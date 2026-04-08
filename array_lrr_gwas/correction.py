"""Batch-effect correction via PC residualization.

The workflow is:

1. Identify *high-quality (HQ)* samples (e.g. low LRR-SD, high call rate).
2. Subset markers using :mod:`array_lrr_gwas.subsetting`.
3. Decompose the (markers × HQ-samples) sub-matrix to obtain batch PCs.
4. **Extrapolate** PCs to remaining (LQ) samples by projecting them onto the
   loadings estimated from HQ data.
5. Regress the batch PCs out of the full LRR matrix to produce corrected
   values.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from array_lrr_gwas.decomposition import decompose, DecompCallable
from array_lrr_gwas.select_k import select_k_mp
from array_lrr_gwas.subsetting import subset_markers

logger = logging.getLogger(__name__)


def classify_samples(
    lrr: NDArray[np.floating],
    max_lrr_sd: float = 0.35,
    min_call_rate: float = 0.95,
) -> NDArray[np.bool_]:
    """Return a boolean mask identifying high-quality samples.

    Parameters
    ----------
    lrr : ndarray, shape (n_markers, n_samples)
    max_lrr_sd : float
        Maximum per-sample standard deviation of LRR to be considered HQ.
    min_call_rate : float
        Minimum fraction of non-missing LRR values per sample.

    Returns
    -------
    hq_mask : ndarray of bool, shape (n_samples,)
    """
    n_markers = lrr.shape[0]
    sd = np.nanstd(lrr, axis=0)
    cr = np.sum(~np.isnan(lrr), axis=0) / n_markers
    return (sd <= max_lrr_sd) & (cr >= min_call_rate)


def extrapolate_pcs(
    lrr_lq: NDArray[np.floating],
    row_means: NDArray[np.floating],
    U: NDArray[np.floating],
    s: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Project low-quality samples onto the PC space of HQ samples.

    Parameters
    ----------
    lrr_lq : ndarray, shape (n_markers, n_lq_samples)
        LRR values for LQ samples (same marker subset as used for decomposition).
    row_means : ndarray, shape (n_markers, 1)
        Per-marker means computed from HQ samples (used for centring).
    U : ndarray, shape (n_markers, k)
        Left singular vectors (marker loadings) from the HQ decomposition.
    s : ndarray, shape (k,)
        Singular values from the HQ decomposition.

    Returns
    -------
    Vt_lq : ndarray, shape (k, n_lq_samples)
        Estimated right singular vectors (PC scores) for LQ samples.
    """
    centred = lrr_lq - row_means
    centred[np.isnan(centred)] = 0.0
    # V_lq = X_lq^T @ U @ diag(1/s)
    Vt_lq = np.diag(1.0 / s) @ U.T @ centred
    return Vt_lq


def residualize(
    lrr: NDArray[np.floating],
    U: NDArray[np.floating],
    s: NDArray[np.floating],
    Vt: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Remove the first *k* principal components from the LRR matrix.

    Parameters
    ----------
    lrr : ndarray, shape (n_markers, n_samples)
        Original (full) LRR matrix.
    U : ndarray, shape (n_markers, k)
        Marker loadings (from the marker-subsetted HQ decomposition,
        but expanded back to the full marker set – see :func:`correct_lrr`).
    s : ndarray, shape (k,)
    Vt : ndarray, shape (k, n_samples)

    Returns
    -------
    corrected : ndarray, shape (n_markers, n_samples)
        LRR with batch effects regressed out.
    """
    batch = U @ np.diag(s) @ Vt
    corrected = lrr - batch
    return corrected


def correct_lrr(
    lrr: NDArray[np.floating],
    positions: NDArray[np.intp] | None = None,
    chromosomes: NDArray | None = None,
    *,
    k: int | None = None,
    n_components: int | None = None,
    max_lrr_sd: float = 0.35,
    min_sample_call_rate: float = 0.95,
    min_marker_call_rate: float = 0.95,
    min_var: float = 0.001,
    max_var: float | None = None,
    exclude_regions: dict[str, list[tuple[int, int]]] | None = None,
    backend: str | DecompCallable = "rsvd",
    upstream_qc_mask: NDArray[np.bool_] | None = None,
    audit: object | None = None,
    variant_ids: list[str] | None = None,
    sample_ids: list[str] | None = None,
) -> tuple[NDArray[np.floating], dict]:
    """End-to-end batch-effect correction for an LRR matrix.

    Parameters
    ----------
    lrr : ndarray, shape (n_markers, n_samples)
        Input LRR values (may contain ``NaN`` for missing calls).
    positions, chromosomes : optional arrays for genomic-complexity filtering.
    k : int or None
        Number of batch components to remove.  If ``None``, chosen
        automatically via the Marchenko–Pastur heuristic.
    n_components : int or None
        Number of components to compute in the pilot truncated decomposition
        used for automatic ``k`` selection. If ``None``, defaults to 5% of HQ
        sample count (minimum 1, capped by feasibility). Ignored when ``k`` is
        provided explicitly.
    max_lrr_sd : float
        HQ-sample threshold on per-sample LRR standard deviation.
    min_sample_call_rate : float
        HQ-sample threshold on per-sample call rate.
    min_marker_call_rate : float
        Marker call-rate threshold for subsetting.
    min_var, max_var : float
        Marker variance thresholds.
    exclude_regions : see :func:`~array_lrr_gwas.subsetting.complexity_mask`.
    backend : decomposition backend (see :func:`~array_lrr_gwas.decomposition.decompose`).
    upstream_qc_mask : ndarray of bool or None
        Pre-computed upstream variant QC mask (e.g. from
        :func:`~array_lrr_gwas.variant_qc.variant_qc_mask` with
        ``require_call_rate=True, require_hwe=True, require_maf=False``).
        When provided, the mask is AND-ed with the subsetting filters.
    audit : AuditLogger or None
        Optional :class:`~array_lrr_gwas.audit.AuditLogger` instance.
        When provided, per-filter marker subsetting and sample
        classification are recorded in the audit trail.
    variant_ids : list of str or None
        Variant ID strings aligned to the rows of *lrr*.  Required when
        *audit* is not ``None`` so that per-marker audit IDs can be recorded.
    sample_ids : list of str or None
        Sample ID strings aligned to the columns of *lrr*.  Required when
        *audit* is not ``None`` so that per-sample HQ/LQ classification
        can be recorded.

    Returns
    -------
    corrected : ndarray, shape (n_markers, n_samples)
        Batch-corrected LRR values.
    info : dict
        Metadata about the correction.  Keys:

        ``k``
            Number of PCs used for correction (may be fewer than
            ``n_components_computed`` when using auto selection).
        ``n_components_computed``
            Total number of PCs that were decomposed (= ``pilot_k`` in auto
            mode, = ``k`` when *k* is provided explicitly).
        ``singular_values``
            All *n_components_computed* singular values in descending order.
        ``sample_scores``
            PC scores for all samples, shape *(n_components_computed, n_samples)*.
        ``marker_loadings``
            Marker loadings, shape *(n_markers_subset, n_components_computed)*.
        ``marker_mask``
            Boolean mask of markers used in the decomposition.
        ``hq_sample_mask``
            Boolean mask of HQ samples.
        ``n_hq_samples``, ``n_markers_used``, ``backend``
            Scalar summary statistics.
    """
    # 1. Classify samples
    hq_mask = classify_samples(
        lrr, max_lrr_sd=max_lrr_sd, min_call_rate=min_sample_call_rate
    )
    n_hq = int(np.sum(hq_mask))
    n_total_samples = lrr.shape[1]
    logger.info(
        "Sample classification: %d / %d HQ (max_lrr_sd=%.4f, "
        "min_call_rate=%.4f), %d LQ",
        n_hq, n_total_samples, max_lrr_sd, min_sample_call_rate,
        n_total_samples - n_hq,
    )
    if not np.any(hq_mask):
        raise ValueError(
            "No samples passed the HQ threshold – relax max_lrr_sd / "
            "min_sample_call_rate or check input data."
        )

    # Record sample HQ/LQ classification in the audit trail
    if audit is not None and sample_ids is not None:
        _hq_ids = [sample_ids[i] for i in range(n_total_samples) if hq_mask[i]]
        _lq_excluded = {
            sample_ids[i]: "low_quality"
            for i in range(n_total_samples) if not hq_mask[i]
        }
        audit.record(
            stage="correction_sample_qc",
            id_type="sample",
            included=_hq_ids,
            excluded=_lq_excluded,
        )

    # 2. Subset markers
    marker_mask = subset_markers(
        lrr,
        positions=positions,
        chromosomes=chromosomes,
        min_call_rate=min_marker_call_rate,
        min_var=min_var,
        max_var=max_var,
        exclude_regions=exclude_regions,
        upstream_qc_mask=upstream_qc_mask,
        audit=audit,
        variant_ids=variant_ids,
    )
    if not np.any(marker_mask):
        raise ValueError(
            "No markers passed QC – relax filter thresholds."
        )

    # 3. Decompose HQ sub-matrix
    sub = lrr[np.ix_(marker_mask, hq_mask)]
    # Centre per marker row for later extrapolation
    row_means = np.nanmean(sub, axis=1, keepdims=True)

    # Determine k
    max_possible_k = min(sub.shape) - 1
    if max_possible_k < 1:
        raise ValueError(
            "Sub-matrix too small for decomposition after filtering."
        )
    if k is None:
        if n_components is None:
            pilot_k = max(1, int(np.ceil(0.05 * sub.shape[1])))
        else:
            if n_components < 1:
                raise ValueError("n_components must be >= 1.")
            pilot_k = n_components
        pilot_k = min(max_possible_k, pilot_k)
        logger.info(
            "Computing %d pilot PCs (%.0f%% of %d HQ samples, capped at %d)",
            pilot_k,
            100 * pilot_k / sub.shape[1],
            sub.shape[1],
            max_possible_k,
        )
        U_full_pilot, s_pilot, Vt_hq_pilot = decompose(sub, pilot_k, backend=backend)
        k = select_k_mp(s_pilot, sub.shape[0], sub.shape[1])
        k = min(k, max_possible_k)
        logger.info(
            "Marchenko-Pastur selected k=%d from %d computed PCs",
            k,
            pilot_k,
        )
        # Use the already-computed decomposition; slice first k components for correction
        n_computed = pilot_k
        U, s, Vt_hq = U_full_pilot[:, :k], s_pilot[:k], Vt_hq_pilot[:k, :]
        U_all, s_all, Vt_hq_all = U_full_pilot, s_pilot, Vt_hq_pilot
    else:
        if k > max_possible_k:
            raise ValueError(
                f"k={k} exceeds max feasible components ({max_possible_k})."
            )
        logger.info("Using explicit k=%d PCs for correction", k)
        n_computed = k
        U, s, Vt_hq = decompose(sub, k, backend=backend)
        U_all, s_all, Vt_hq_all = U, s, Vt_hq

    # 4. Extrapolate PCs to LQ samples (using all computed components)
    lq_mask = ~hq_mask
    n_lq = int(np.sum(lq_mask))
    if n_lq > 0:
        Vt_lq_all = extrapolate_pcs(
            lrr[np.ix_(marker_mask, lq_mask)], row_means, U_all, s_all
        )
        # Assemble full Vt for all computed components (n_computed × n_samples)
        Vt_full_all = np.empty((n_computed, lrr.shape[1]), dtype=np.float64)
        Vt_full_all[:, hq_mask] = Vt_hq_all
        Vt_full_all[:, lq_mask] = Vt_lq_all
    else:
        Vt_full_all = Vt_hq_all

    # 5. Expand U back to full marker dimension (only k components for residualization)
    U_full = np.zeros((lrr.shape[0], k), dtype=np.float64)
    U_full[marker_mask, :] = U

    # 6. Residualize using only the k selected components
    corrected = residualize(lrr, U_full, s, Vt_full_all[:k, :])

    info = {
        "k": k,
        "n_components_computed": n_computed,
        "singular_values": s_all,
        "sample_scores": Vt_full_all,
        "marker_loadings": U_all,
        "marker_mask": marker_mask,
        "hq_sample_mask": hq_mask,
        "n_hq_samples": int(np.sum(hq_mask)),
        "n_markers_used": int(np.sum(marker_mask)),
        "backend": backend if isinstance(backend, str) else "custom",
    }
    return corrected, info
