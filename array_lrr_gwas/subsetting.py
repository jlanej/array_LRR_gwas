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

import warnings
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


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
    present = np.sum(~np.isnan(lrr), axis=1)
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

    Returns
    -------
    mask : ndarray of bool, shape (n_markers,)
        ``True`` for markers passing **all** enabled filters.
    """
    mask = call_rate_mask(lrr, min_call_rate=min_call_rate)
    mask &= variance_mask(lrr, min_var=min_var, max_var=max_var)
    if chromosomes is not None:
        if autosomes_only:
            mask &= autosome_mask(chromosomes)
        if positions is not None:
            mask &= complexity_mask(positions, chromosomes, exclude_regions)
    if upstream_qc_mask is not None:
        mask &= upstream_qc_mask
    return mask
