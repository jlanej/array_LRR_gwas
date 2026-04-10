"""Batch-effect correction via PC residualization.

The workflow is:

1. Identify *high-quality (HQ)* samples (e.g. low LRR-SD, high call rate).
2. Subset markers using :mod:`array_lrr_gwas.subsetting`.
3. Decompose the (markers × HQ-samples) sub-matrix to obtain batch PCs.
4. **Extrapolate** PCs to remaining (LQ) samples by projecting them onto the
   loadings estimated from HQ data.
5. Precompute the QR decomposition of the global PC design matrix.
6. Regress the batch PCs out of **all** markers via streaming, chunked QR
   regression with robust per-marker missing-data handling.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import qr as _scipy_qr
from tqdm.auto import tqdm

from array_lrr_gwas.decomposition import decompose, DecompCallable
from array_lrr_gwas.select_k import select_k_mp
from array_lrr_gwas.subsetting import autosome_mask, subset_markers

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
    # Use isfinite to exclude both NaN and inf values.  np.nanstd returns NaN
    # when a column contains inf (inf - mean = NaN), and inf values should not
    # count toward the call rate since they are not valid measurements.
    finite_mask = np.isfinite(lrr)
    lrr_finite = np.where(finite_mask, lrr, np.nan)
    sd = np.nanstd(lrr_finite, axis=0)
    cr = np.sum(finite_mask, axis=0) / n_markers
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


# ---------------------------------------------------------------------------
# QR-based streaming PC regression
# ---------------------------------------------------------------------------

def qr_precompute(
    Vt_k: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Precompute the QR decomposition of the PC-score design matrix.

    The design matrix is ``X = Vt_k.T`` with shape *(n_samples, k)*.
    A thin (economy) QR factorisation ``X = Q R`` is computed once and
    reused for every marker during streaming correction.

    Parameters
    ----------
    Vt_k : ndarray, shape (k, n_samples)
        PC scores for all samples (the first *k* right singular vectors
        from the RSVD decomposition).

    Returns
    -------
    Q : ndarray, shape (n_samples, k)
        Orthonormal basis for the column space of *X*.
    R : ndarray, shape (k, k)
        Upper-triangular factor.
    X : ndarray, shape (n_samples, k)
        The design matrix ``Vt_k.T`` (kept for the per-marker fallback
        path when missing data requires re-solving).
    """
    X = np.ascontiguousarray(Vt_k.T, dtype=np.float64)
    Q, R = _scipy_qr(X, mode="economic")
    return Q, R, X


def _correct_chunk_qr(
    chunk: NDArray[np.floating],
    Q: NDArray[np.floating],
    X: NDArray[np.floating],
    min_valid_frac: float = 0.5,
) -> NDArray[np.floating]:
    """Correct a chunk of markers using precomputed QR regression.

    For markers with **no** missing data the fast path is used::

        corrected = y − Q (Qᵀ y)

    For markers that contain ``NaN`` values, the regression is re-solved
    on the valid (finite) subset of samples via :func:`numpy.linalg.lstsq`.
    Missing positions are left as ``NaN`` in the output.

    If fewer than ``max(k + 1, min_valid_frac × n_samples)`` finite
    values are available, the marker is returned **uncorrected**.

    Parameters
    ----------
    chunk : ndarray, shape (n_chunk, n_samples)
        LRR values for the current marker chunk.
    Q : ndarray, shape (n_samples, k)
        Orthonormal Q factor from :func:`qr_precompute`.
    X : ndarray, shape (n_samples, k)
        Full design matrix from :func:`qr_precompute` (used only for the
        per-marker NaN fallback).
    min_valid_frac : float
        Minimum fraction of finite samples required for correction.

    Returns
    -------
    corrected : ndarray, shape (n_chunk, n_samples)
    n_skipped : int
        Number of markers that were left uncorrected due to insufficient
        valid data.
    """
    n_chunk, n_samples = chunk.shape
    k = Q.shape[1]
    corrected = chunk.copy()
    n_skipped = 0

    # Identify markers with / without non-finite values (NaN or inf)
    has_nonfinite = ~np.all(np.isfinite(chunk), axis=1)

    # --- fast path (all finite) – vectorised over the clean rows --------
    no_nan_idx = np.flatnonzero(~has_nonfinite)
    if len(no_nan_idx) > 0:
        Y = chunk[no_nan_idx]           # (n_clean, n_samples)
        coeffs = Y @ Q                  # (n_clean, k)
        proj = coeffs @ Q.T             # (n_clean, n_samples)
        corrected[no_nan_idx] = Y - proj

    # --- slow path (per-marker with non-finite values) ------------------
    min_valid = max(k + 1, int(np.ceil(min_valid_frac * n_samples)))
    for idx in np.flatnonzero(has_nonfinite):
        y = chunk[idx]
        valid = np.isfinite(y)
        n_valid = int(valid.sum())

        if n_valid < min_valid:
            # Not enough data – leave uncorrected
            n_skipped += 1
            continue

        X_v = X[valid]
        y_v = y[valid]
        beta, _, _, _ = np.linalg.lstsq(X_v, y_v, rcond=None)
        corrected[idx, valid] = y_v - X_v @ beta

    return corrected, n_skipped


def residualize_qr(
    lrr: NDArray[np.floating],
    Vt_k: NDArray[np.floating],
    *,
    chunk_size: int = 5000,
    min_valid_frac: float = 0.5,
) -> NDArray[np.floating]:
    """Streaming QR-based PC regression for **all** markers.

    The PC-score design matrix ``X = Vt_k.T`` is factorised once via
    :func:`qr_precompute`.  Markers are then processed in chunks of
    *chunk_size* rows so that at most two chunks are resident in memory
    at any time.

    Parameters
    ----------
    lrr : ndarray, shape (n_markers, n_samples)
        Original LRR matrix (may contain ``NaN``).
    Vt_k : ndarray, shape (k, n_samples)
        PC scores for all samples (first *k* components).
    chunk_size : int
        Number of marker rows processed per iteration.
    min_valid_frac : float
        Minimum fraction of finite samples required per marker; markers
        below this threshold are returned uncorrected.

    Returns
    -------
    corrected : ndarray, shape (n_markers, n_samples)
    """
    Q, R, X = qr_precompute(Vt_k)
    n_markers, n_samples = lrr.shape
    k = Vt_k.shape[0]
    n_chunks = (n_markers + chunk_size - 1) // chunk_size
    corrected = np.empty_like(lrr)

    logger.info(
        "PC correction: regressing %d PCs out of %d markers × %d samples "
        "in %d chunk(s) of %d",
        k, n_markers, n_samples, n_chunks, chunk_size,
    )

    n_skipped = 0
    with tqdm(
        total=n_markers,
        desc="PC regression",
        unit="marker",
        leave=False,
        dynamic_ncols=True,
    ) as pbar:
        for chunk_idx, start in enumerate(range(0, n_markers, chunk_size)):
            end = min(start + chunk_size, n_markers)
            chunk = lrr[start:end]
            corrected_chunk, chunk_skipped = _correct_chunk_qr(
                chunk, Q, X, min_valid_frac=min_valid_frac,
            )
            corrected[start:end] = corrected_chunk
            n_skipped += chunk_skipped
            pbar.update(end - start)
            pbar.set_postfix(
                chunk=f"{chunk_idx + 1}/{n_chunks}",
                skipped=n_skipped,
                refresh=False,
            )
            logger.debug(
                "PC correction chunk %d/%d: markers %d–%d (%d skipped so far)",
                chunk_idx + 1, n_chunks, start, end - 1, n_skipped,
            )

    if n_skipped > 0:
        logger.info(
            "QR regression: %d / %d markers skipped (insufficient valid data)",
            n_skipped,
            n_markers,
        )
    else:
        logger.info("PC correction complete: all %d markers corrected", n_markers)

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
    max_ram_gb: float | None = None,
    chunk_size: int = 5000,
    min_valid_frac: float = 0.5,
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
    max_ram_gb : float or None
        Maximum RAM in GB available for the RSVD decomposition step.
        When the QC-passing marker set would exceed this budget, markers
        are deterministically subsampled to fit using genome-uniform
        sampling.  ``None`` disables the budget (no subsampling).
        ``0`` also disables subsampling.
    chunk_size : int
        Number of marker rows processed per streaming QR-regression
        iteration.  Controls peak memory: at most two chunks are resident
        simultaneously.
    min_valid_frac : float
        Minimum fraction of finite (non-``NaN``) samples required per
        marker for QR regression.  Markers below this threshold (or with
        fewer than ``k + 1`` valid values) are left uncorrected.

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
    # 0. Log upfront matrix dimensions and per-GB density estimate.
    n_total_markers, n_total_samples = lrr.shape
    bytes_per_element = lrr.dtype.itemsize if lrr.dtype.itemsize > 0 else 8
    matrix_gb = (n_total_markers * n_total_samples * bytes_per_element) / 1024**3
    variants_per_gb = (n_total_markers * n_total_samples) / max(matrix_gb, 1e-12)
    logger.info(
        "Input matrix: %d variants × %d samples (%.2f GB @ %d B/element; "
        "%.3g variant×sample per GB)",
        n_total_markers, n_total_samples, matrix_gb,
        bytes_per_element, variants_per_gb,
    )

    # Pipeline-level progress bar.  We start with a fixed set of core stages
    # and dynamically extend the total if optional stages run.
    _core_stages = 5  # sample QC, marker subset, SVD load, decompose, PC regression
    pbar = tqdm(
        total=_core_stages,
        desc="correct_lrr",
        unit="step",
        leave=True,
        dynamic_ncols=True,
    )

    # 1. Classify samples using only autosomal markers for LRR_SD and callrate.
    # Non-autosomal (sex chromosome, MT) intensity signals encode biological sex
    # rather than technical noise and must not contribute to sample QC metrics.
    pbar.set_description("sample QC")
    if chromosomes is not None:
        auto_m = autosome_mask(chromosomes)
        lrr_for_qc = lrr[auto_m]
    else:
        lrr_for_qc = lrr
    hq_mask = classify_samples(
        lrr_for_qc, max_lrr_sd=max_lrr_sd, min_call_rate=min_sample_call_rate
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
        pbar.close()
        raise ValueError(
            "No samples passed the HQ threshold – relax max_lrr_sd / "
            "min_sample_call_rate or check input data."
        )
    pbar.set_postfix(HQ=n_hq, LQ=n_total_samples - n_hq, refresh=False)
    pbar.update(1)

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
    pbar.set_description("marker subsetting")
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
        pbar.close()
        raise ValueError(
            "No markers passed QC – relax filter thresholds."
        )
    pbar.set_postfix(markers=int(marker_mask.sum()), refresh=False)
    pbar.update(1)

    # 2b. Optionally subsample markers to fit within the RAM budget
    n_candidates_pre_budget = int(marker_mask.sum())
    budget = None
    if max_ram_gb is not None and max_ram_gb > 0:
        from array_lrr_gwas.decomposition import estimate_rsvd_marker_budget
        from array_lrr_gwas.subsetting import subsample_markers_uniform

        pbar.total += 1
        pbar.set_description("budget subsampling")
        max_ram_bytes = int(max_ram_gb * 1024**3)
        k_for_budget = max(1, int(np.ceil(0.05 * n_hq))) if k is None else k
        budget = estimate_rsvd_marker_budget(
            n_hq, k_for_budget, max_ram_bytes=max_ram_bytes
        )
        n_candidates = int(marker_mask.sum())
        if n_candidates > budget:
            logger.warning(
                "RSVD marker budget exceeded: %d markers > %d budget "
                "(%.1f GB limit). Subsampling to %d markers "
                "with genome-uniform strategy.",
                n_candidates, budget, max_ram_gb, budget,
            )
            if positions is not None and chromosomes is not None:
                candidate_idx = np.flatnonzero(marker_mask)
                selected_idx = subsample_markers_uniform(
                    candidate_idx,
                    chromosomes=chromosomes,
                    positions=positions,
                    target_n=budget,
                    random_state=0,
                )
                new_mask = np.zeros(lrr.shape[0], dtype=bool)
                new_mask[selected_idx] = True
                marker_mask = new_mask
            else:
                # Without genomic coordinates, fall back to uniform random
                candidate_idx = np.flatnonzero(marker_mask)
                rng = np.random.default_rng(0)
                chosen = rng.choice(candidate_idx, size=budget, replace=False)
                new_mask = np.zeros(lrr.shape[0], dtype=bool)
                new_mask[chosen] = True
                marker_mask = new_mask
            logger.info(
                "Subsampled to %d markers for RSVD (seed=0).",
                int(marker_mask.sum()),
            )
        else:
            logger.info(
                "RSVD marker count %d within %.1f GB budget (%d markers); "
                "no subsampling needed.",
                n_candidates, max_ram_gb, budget,
            )
        pbar.set_postfix(markers=int(marker_mask.sum()), budget=budget, refresh=False)
        pbar.update(1)

    # 3. Decompose HQ sub-matrix
    pbar.set_description("loading SVD matrix")
    sub = lrr[np.ix_(marker_mask, hq_mask)]
    n_sub_markers, n_sub_samples = sub.shape
    sub_gb = (n_sub_markers * n_sub_samples * 8) / 1024**3
    logger.info(
        "Loading SVD sub-matrix: %d variants × %d HQ samples (%.2f GB) "
        "from %d total variants",
        n_sub_markers, n_sub_samples, sub_gb, n_total_markers,
    )
    # Centre per marker row for later extrapolation
    row_means = np.nanmean(sub, axis=1, keepdims=True)
    pbar.set_postfix(
        svd_shape=f"{n_sub_markers}×{n_sub_samples}",
        svd_gb=f"{sub_gb:.2f}",
        refresh=False,
    )
    pbar.update(1)

    # Determine k
    max_possible_k = min(sub.shape) - 1
    if max_possible_k < 1:
        pbar.close()
        raise ValueError(
            "Sub-matrix too small for decomposition after filtering."
        )
    pbar.set_description("SVD decomposition")
    if k is None:
        if n_components is None:
            pilot_k = max(1, int(np.ceil(0.05 * sub.shape[1])))
        else:
            if n_components < 1:
                pbar.close()
                raise ValueError("n_components must be >= 1.")
            pilot_k = n_components
        pilot_k = min(max_possible_k, pilot_k)
        logger.info(
            "Computing %d pilot PCs (%.0f%% of %d HQ samples, capped at %d)",
            pilot_k,
            100 * pilot_k / max(1, sub.shape[1]),
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
            pbar.close()
            raise ValueError(
                f"k={k} exceeds max feasible components ({max_possible_k})."
            )
        logger.info("Using explicit k=%d PCs for correction", k)
        n_computed = k
        U, s, Vt_hq = decompose(sub, k, backend=backend)
        U_all, s_all, Vt_hq_all = U, s, Vt_hq
    pbar.set_postfix(k=k, refresh=False)
    pbar.update(1)

    # 4. Extrapolate PCs to LQ samples (using all computed components)
    lq_mask = ~hq_mask
    n_lq = int(np.sum(lq_mask))
    if n_lq > 0:
        pbar.total += 1
        pbar.set_description("PC extrapolation")
        Vt_lq_all = extrapolate_pcs(
            lrr[np.ix_(marker_mask, lq_mask)], row_means, U_all, s_all
        )
        # Assemble full Vt for all computed components (n_computed × n_samples)
        Vt_full_all = np.empty((n_computed, lrr.shape[1]), dtype=np.float64)
        Vt_full_all[:, hq_mask] = Vt_hq_all
        Vt_full_all[:, lq_mask] = Vt_lq_all
        pbar.set_postfix(k=k, LQ_projected=n_lq, refresh=False)
        pbar.update(1)
    else:
        Vt_full_all = Vt_hq_all

    # 5-6. Regress PCs out of ALL markers via streaming QR decomposition.
    # Unlike the old SVD-based residualize() which only corrected the
    # marker subset used for decomposition, QR regression corrects every
    # marker by regressing its LRR values against the global PC scores.
    pbar.set_description("PC regression")
    corrected = residualize_qr(
        lrr,
        Vt_full_all[:k, :],
        chunk_size=chunk_size,
        min_valid_frac=min_valid_frac,
    )
    pbar.set_postfix(k=k, markers_corrected=n_total_markers, refresh=False)
    pbar.update(1)
    pbar.set_description("done")
    pbar.close()

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
        "rsvd_subsampled": (
            max_ram_gb is not None
            and max_ram_gb > 0
            and int(marker_mask.sum()) < n_candidates_pre_budget
        ),
        "rsvd_marker_budget": budget,
    }
    return corrected, info
