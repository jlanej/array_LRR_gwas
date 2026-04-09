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


def subsample_markers_uniform(
    candidate_indices: NDArray[np.intp],
    chromosomes: NDArray,
    positions: NDArray[np.intp],
    target_n: int,
    *,
    n_bins_per_chrom: int = 1000,
    random_state: int | np.random.RandomState | None = 0,
) -> NDArray[np.intp]:
    """Deterministically subsample *target_n* markers with genome-uniform
    representation.

    Parameters
    ----------
    candidate_indices : ndarray of int, shape (n_candidates,)
        Row indices into the original full LRR matrix that passed
        all upstream QC filters.
    chromosomes : ndarray of str, shape (n_total_markers,)
        Chromosome label for every marker row (not just candidates).
    positions : ndarray of int, shape (n_total_markers,)
        Base-pair position for every marker row.
    target_n : int
        Desired number of markers after subsampling.
    n_bins_per_chrom : int
        Number of equal-width position bins per chromosome.
    random_state : int, RandomState, or None
        Master seed for reproducibility.

    Returns
    -------
    selected_indices : ndarray of int, shape (≤ target_n,)
        Sorted row indices of the selected markers.
    """
    candidate_indices = np.asarray(candidate_indices)
    n_candidates = len(candidate_indices)
    if target_n >= n_candidates:
        return np.sort(candidate_indices)

    chromosomes = np.asarray(chromosomes, dtype=str)
    positions = np.asarray(positions)

    # Resolve master seed to an integer for deterministic per-bin seeding
    if isinstance(random_state, np.random.RandomState):
        master_seed = random_state.randint(0, 2**31)
    elif random_state is None:
        master_seed = 0
    else:
        master_seed = int(random_state)

    # Assign each candidate to a (chromosome, bin) pair
    cand_chroms = chromosomes[candidate_indices]
    cand_pos = positions[candidate_indices]
    unique_chroms = np.unique(cand_chroms)

    # Build bin assignments
    bin_keys: list[tuple[int, int]] = []  # (chrom_idx, bin_idx) per candidate
    chrom_to_idx = {c: i for i, c in enumerate(unique_chroms)}

    for chrom in unique_chroms:
        chrom_mask = cand_chroms == chrom
        chrom_positions = cand_pos[chrom_mask]
        if len(chrom_positions) == 0:
            continue
        pmin, pmax = chrom_positions.min(), chrom_positions.max()
        if pmax == pmin:
            # All same position — single bin
            for _ in range(int(chrom_mask.sum())):
                bin_keys.append((chrom_to_idx[chrom], 0))
        else:
            bin_width = (pmax - pmin + 1) / n_bins_per_chrom
            bins = np.clip(
                ((chrom_positions - pmin) / bin_width).astype(int),
                0,
                n_bins_per_chrom - 1,
            )
            for b in bins:
                bin_keys.append((chrom_to_idx[chrom], int(b)))

    # Group candidates by bin
    from collections import defaultdict
    bin_to_candidates: dict[tuple[int, int], list[int]] = defaultdict(list)
    # Iterate in a consistent order: sort candidates by chrom, then position
    # We need to map bin_keys back to candidate_indices
    # Build the mapping by iterating chromosomes in order
    idx_ptr = 0
    for chrom in unique_chroms:
        chrom_mask = cand_chroms == chrom
        chrom_cand_indices = candidate_indices[chrom_mask]
        chrom_positions = cand_pos[chrom_mask]
        if len(chrom_cand_indices) == 0:
            continue
        pmin, pmax = chrom_positions.min(), chrom_positions.max()
        if pmax == pmin:
            bins = np.zeros(len(chrom_cand_indices), dtype=int)
        else:
            bin_width = (pmax - pmin + 1) / n_bins_per_chrom
            bins = np.clip(
                ((chrom_positions - pmin) / bin_width).astype(int),
                0,
                n_bins_per_chrom - 1,
            )
        ci = chrom_to_idx[chrom]
        for j, cand_idx in enumerate(chrom_cand_indices):
            bin_to_candidates[(ci, int(bins[j]))].append(int(cand_idx))

    # Pass 1: compute per-bin quotas proportional to bin size
    bin_list = sorted(bin_to_candidates.keys())
    bin_sizes = np.array([len(bin_to_candidates[b]) for b in bin_list])
    total = bin_sizes.sum()
    # Proportional allocation with floor rounding
    raw_quotas = (bin_sizes / total) * target_n
    quotas = np.floor(raw_quotas).astype(int)
    # Distribute remaining slots to bins with largest fractional remainder.
    # Break ties deterministically using a seeded random jitter so that
    # equal-size bins across chromosomes are treated fairly.
    remainder = target_n - quotas.sum()
    if remainder > 0:
        fractional = raw_quotas - quotas
        rng_tie = np.random.default_rng(master_seed + 7)
        tie_break = rng_tie.random(len(bin_list))
        sort_key = fractional + tie_break * 1e-10
        top_bins = np.argsort(-sort_key, kind="stable")[:remainder]
        quotas[top_bins] += 1
    # Ensure no bin quota exceeds its size
    quotas = np.minimum(quotas, bin_sizes)

    # Pass 2: select markers within each bin
    selected = []
    for i, bkey in enumerate(bin_list):
        q = quotas[i]
        if q == 0:
            continue
        candidates_in_bin = np.array(bin_to_candidates[bkey])
        # Sort for deterministic ordering before sampling
        candidates_in_bin.sort()
        if q >= len(candidates_in_bin):
            selected.extend(candidates_in_bin.tolist())
        else:
            # Deterministic per-bin seed from (master_seed, chrom_idx, bin_idx)
            bin_seed = (master_seed * 1_000_003 + bkey[0] * 1_000_033 + bkey[1]) % (2**31)
            rng = np.random.default_rng(bin_seed)
            chosen = rng.choice(candidates_in_bin, size=q, replace=False)
            selected.extend(chosen.tolist())

    return np.sort(np.array(selected, dtype=np.intp))
