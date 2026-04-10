"""Marker subsetting for batch effect correction.

Provides filters to select high-quality markers from an LRR matrix based on
scientifically defensible criteria:
- **Autosomal filter**: restricts to autosomal chromosomes, excluding sex
  chromosomes (X, Y) and mitochondrial markers (MT/M) that would otherwise
  drive PCA to capture sex-based rather than technical batch variation.
- **Call-rate filter**: removes markers with excessive missingness.
- **Variance filter**: removes markers with near-zero variance (uninformative)
  or extremely high variance (likely artefactual).
- **Genomic-complexity filter**: optionally excludes markers in regions of
  known low-complexity or segmental duplication.
- **Every-Nth selection**: when a RAM budget is active, selects every Nth
  QC-passing marker to keep memory usage within bounds.
"""

from __future__ import annotations

import logging
import warnings
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def call_rate_mask(
    lrr: NDArray[np.floating],
    min_call_rate: float = 0.95,
) -> NDArray[np.bool_]:
    """Return a boolean mask for markers exceeding a call-rate threshold.

    Parameters
    ----------
    lrr : ndarray, shape (n_markers, n_samples)
        LRR matrix where ``np.nan`` denotes a missing call.
    min_call_rate : float
        Minimum fraction of non-missing values required to keep a marker.

    Returns
    -------
    mask : ndarray of bool, shape (n_markers,)
        ``True`` for markers that pass the call-rate filter.
    """
    if lrr.ndim != 2:
        raise ValueError("lrr must be a 2-D array (markers × samples)")
    n_samples = lrr.shape[1]
    # Count finite values per marker; inf is not a valid measurement.
    present = np.sum(np.isfinite(lrr), axis=1)
    return (present / n_samples) >= min_call_rate


def variance_mask(
    lrr: NDArray[np.floating],
    min_var: float = 0.001,
    max_var: float | None = None,
) -> NDArray[np.bool_]:
    """Return a boolean mask for markers within an acceptable variance range.

    Markers with near-zero variance are uninformative; markers with extreme
    variance are likely artefactual.

    Parameters
    ----------
    lrr : ndarray, shape (n_markers, n_samples)
    min_var : float
        Minimum per-marker variance (across samples).
    max_var : float or None
        Maximum per-marker variance.  ``None`` disables the upper bound.

    Returns
    -------
    mask : ndarray of bool, shape (n_markers,)
    """
    if lrr.ndim != 2:
        raise ValueError("lrr must be a 2-D array (markers × samples)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        var = np.nanvar(lrr, axis=1)
    # All-NaN rows produce NaN variance → treat as failing the filter.
    mask = var >= min_var  # NaN >= x is False
    if max_var is not None:
        mask &= var <= max_var
    return mask


# Non-autosomal chromosome labels (chr-prefixed and bare).
_NON_AUTOSOMAL: frozenset[str] = frozenset({
    "chrX", "chrY", "chrM", "chrMT",
    "X", "Y", "M", "MT",
})


def autosome_mask(
    chromosomes: NDArray | Sequence[str],
) -> NDArray[np.bool_]:
    """Return a boolean mask that keeps only autosomal markers.

    Sex chromosomes (X, Y) and mitochondrial markers (M, MT) are excluded
    because their intensity signals encode biological sex rather than
    technical batch effects, which would confound PCA-based correction.

    Parameters
    ----------
    chromosomes : array-like of str, shape (n_markers,)
        Chromosome label for each marker.

    Returns
    -------
    mask : ndarray of bool, shape (n_markers,)
        ``True`` for autosomal markers.
    """
    chroms = np.asarray(chromosomes, dtype=str)
    return ~np.isin(chroms, list(_NON_AUTOSOMAL))


def complexity_mask(
    positions: NDArray[np.intp],
    chromosomes: NDArray | Sequence[str],
    exclude_regions: dict[str, list[tuple[int, int]]] | None = None,
) -> NDArray[np.bool_]:
    """Exclude markers falling within specified genomic regions.

    This allows removal of markers in centromeres, segmental duplications,
    or other regions of low mappability.

    Parameters
    ----------
    positions : ndarray of int, shape (n_markers,)
        Base-pair position for each marker.
    chromosomes : array-like of str, shape (n_markers,)
        Chromosome label for each marker.
    exclude_regions : dict mapping chromosome -> list of (start, end)
        Genomic intervals to exclude.  If ``None``, no markers are excluded.

    Returns
    -------
    mask : ndarray of bool, shape (n_markers,)
        ``True`` for markers *outside* all excluded regions.
    """
    n = len(positions)
    mask = np.ones(n, dtype=bool)
    if exclude_regions is None:
        return mask
    chroms = np.asarray(chromosomes, dtype=str)
    for chrom, intervals in exclude_regions.items():
        chrom_idx = chroms == chrom
        for start, end in intervals:
            in_region = chrom_idx & (positions >= start) & (positions <= end)
            mask[in_region] = False
    return mask


def subset_markers(
    lrr: NDArray[np.floating],
    positions: NDArray[np.intp] | None = None,
    chromosomes: NDArray | Sequence[str] | None = None,
    min_call_rate: float = 0.95,
    min_var: float = 0.001,
    max_var: float | None = None,
    exclude_regions: dict[str, list[tuple[int, int]]] | None = None,
    autosomes_only: bool = True,
    upstream_qc_mask: NDArray[np.bool_] | None = None,
    audit: object | None = None,
    variant_ids: Sequence[str] | None = None,
) -> NDArray[np.bool_]:
    """Combine QC filters to produce a single marker-keep mask.

    Parameters
    ----------
    lrr : ndarray, shape (n_markers, n_samples)
    positions, chromosomes : optional arrays for complexity filtering.
    min_call_rate, min_var, max_var : QC thresholds (see individual filters).
    exclude_regions : see :func:`complexity_mask`.
    autosomes_only : bool
        If ``True`` (default) and *chromosomes* is provided, non-autosomal
        markers (X, Y, MT) are excluded.  These markers carry sex-linked
        intensity signals that would cause top PCs to capture sex rather
        than technical batch effects.
    upstream_qc_mask : ndarray of bool or None
        Pre-computed upstream variant QC mask (e.g. from
        :func:`~array_lrr_gwas.variant_qc.variant_qc_mask`).  When
        provided, the mask is AND-ed with the other filters.
    audit : AuditLogger or None
        Optional :class:`~array_lrr_gwas.audit.AuditLogger` instance.
        When provided, per-filter audit records are emitted for each
        subsetting step.
    variant_ids : sequence of str or None
        Variant ID strings aligned to the rows of *lrr*.  Required when
        *audit* is not ``None`` so that excluded IDs can be recorded.

    Returns
    -------
    mask : ndarray of bool, shape (n_markers,)
        ``True`` for markers passing **all** enabled filters.
    """
    mask = call_rate_mask(lrr, min_call_rate=min_call_rate)
    n_total = len(mask)
    n_call_rate = int(mask.sum())
    logger.info(
        "Marker subsetting: call-rate filter (≥%.4f): %d / %d pass (%d excluded)",
        min_call_rate, n_call_rate, n_total, n_total - n_call_rate,
    )

    # Collect per-filter exclusion masks for audit trail
    _do_audit = audit is not None and variant_ids is not None
    _vids = list(variant_ids) if variant_ids is not None else None

    if _do_audit:
        _excluded = {
            _vids[i]: "failed_call_rate"
            for i in range(n_total) if not mask[i]
        }
        _included = [_vids[i] for i in range(n_total) if mask[i]]
        audit.record(
            stage="correction_marker_call_rate",
            id_type="marker",
            included=_included,
            excluded=_excluded,
        )

    var_m = variance_mask(lrr, min_var=min_var, max_var=max_var)
    n_var = int(var_m.sum())
    mask &= var_m
    logger.info(
        "Marker subsetting: variance filter (min=%.4f, max=%s): %d / %d pass (%d excluded)",
        min_var, max_var if max_var is not None else "None",
        n_var, n_total, n_total - n_var,
    )

    if _do_audit:
        _excluded = {
            _vids[i]: "failed_variance"
            for i in range(n_total) if not var_m[i]
        }
        _included = [_vids[i] for i in range(n_total) if var_m[i]]
        audit.record(
            stage="correction_marker_variance",
            id_type="marker",
            included=_included,
            excluded=_excluded,
        )

    if chromosomes is not None:
        if autosomes_only:
            auto_m = autosome_mask(chromosomes)
            n_auto = int(auto_m.sum())
            mask &= auto_m
            logger.info(
                "Marker subsetting: autosome filter: %d / %d pass (%d non-autosomal excluded)",
                n_auto, n_total, n_total - n_auto,
            )
            if _do_audit:
                _excluded = {
                    _vids[i]: "non_autosomal"
                    for i in range(n_total) if not auto_m[i]
                }
                _included = [_vids[i] for i in range(n_total) if auto_m[i]]
                audit.record(
                    stage="correction_marker_autosome",
                    id_type="marker",
                    included=_included,
                    excluded=_excluded,
                )
        if positions is not None:
            comp_m = complexity_mask(positions, chromosomes, exclude_regions)
            n_comp = int(comp_m.sum())
            mask &= comp_m
            logger.info(
                "Marker subsetting: complexity-region filter: %d / %d pass (%d in excluded regions)",
                n_comp, n_total, n_total - n_comp,
            )
            if _do_audit:
                _excluded = {
                    _vids[i]: "complexity_region"
                    for i in range(n_total) if not comp_m[i]
                }
                _included = [_vids[i] for i in range(n_total) if comp_m[i]]
                audit.record(
                    stage="correction_marker_complexity",
                    id_type="marker",
                    included=_included,
                    excluded=_excluded,
                )
    if upstream_qc_mask is not None:
        n_qc = int(upstream_qc_mask.sum())
        mask &= upstream_qc_mask
        logger.info(
            "Marker subsetting: upstream variant QC mask: %d / %d pass (%d excluded by QC)",
            n_qc, n_total, n_total - n_qc,
        )

    n_final = int(mask.sum())
    logger.info(
        "Marker subsetting: %d / %d markers pass all filters (%.1f%%)",
        n_final, n_total, 100.0 * n_final / n_total if n_total > 0 else 0.0,
    )
    return mask


def select_every_nth(
    passing_ids: Sequence[str],
    target_n: int,
) -> list[str]:
    """Select approximately *target_n* IDs from *passing_ids* by taking
    every Nth element.

    This deterministic stride-based selection is used when a RAM budget
    requires loading fewer markers than are available.  It preserves the
    original ordering and provides roughly uniform genomic coverage as
    long as the input list is in genomic order (which is the case for
    BCF/VCF-derived variant IDs and QC files).

    Parameters
    ----------
    passing_ids : sequence of str
        Ordered list of variant IDs that passed upstream QC.
    target_n : int
        Desired number of markers to select.  If *target_n* is ≥
        ``len(passing_ids)``, all IDs are returned unchanged.

    Returns
    -------
    selected : list of str
        Selected variant IDs in the same relative order as *passing_ids*.
    """
    n = len(passing_ids)
    if target_n <= 0:
        return []
    if target_n >= n:
        return list(passing_ids)
    step = max(1, n // target_n)
    selected = [passing_ids[i] for i in range(0, n, step)]
    # Trim to exactly target_n if overshoot due to rounding
    return selected[:target_n]
