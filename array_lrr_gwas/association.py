"""GWAS association engine for continuous LRR predictors.

Provides LMM (default), OLS, and logistic regression association scans.
The LMM uses spectral decomposition of a GRM with profile-REML variance
component estimation (FaST-LMM / EMMA approach).

See ``docs/association_engine_design.md`` for the full engine evaluation
and design rationale.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

from tqdm.auto import tqdm

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)


# OLS: number of markers to residualise at once (same concept as LMM chunks).
_OLS_CHUNK_SIZE: int = 10_000

# Logistic IRLS: clamp linear predictor to avoid overflow in exp().
_MAX_ETA: float = 20.0

# LMM: number of markers to rotate into the eigenbasis at once.
# Keeps peak memory bounded for biobank-scale LRR matrices.
_LMM_CHUNK_SIZE: int = 10_000

# LMM heuristic for binary phenotypes treated as continuous.
_LMM_BINARY_CASE_FRAC_WARN_MIN: float = 0.10
_LMM_BINARY_CASE_FRAC_WARN_MAX: float = 0.90

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class AssociationResult:
    """Per-marker association summary statistics."""

    chrom: list[str]
    pos: list[int]
    variant_id: list[str]
    beta: np.ndarray
    se: np.ndarray
    stat: np.ndarray
    p_value: np.ndarray
    n_samples: np.ndarray
    method: str

    def to_records(self) -> list[dict[str, object]]:
        """Convert to a list of dicts (one per variant)."""
        records: list[dict[str, object]] = []
        for i in range(len(self.chrom)):
            records.append({
                "chrom": self.chrom[i],
                "pos": self.pos[i],
                "variant_id": self.variant_id[i],
                "beta": float(self.beta[i]),
                "se": float(self.se[i]),
                "stat": float(self.stat[i]),
                "p_value": float(self.p_value[i]),
                "n_samples": int(self.n_samples[i]),
                "method": self.method,
            })
        return records


# ---------------------------------------------------------------------------
# OLS helpers
# ---------------------------------------------------------------------------

def _ols_scan_complete(
    lrr: np.ndarray,
    phenotype: np.ndarray,
    C: np.ndarray,
    df_cov: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fast vectorised OLS when no LRR values are missing.

    Uses the Frisch-Waugh-Lovell theorem: project covariates out of
    both the phenotype and the LRR rows, then run univariate regression
    on the residuals.

    LRR residualisation is performed in chunks of ``_OLS_CHUNK_SIZE``
    markers to keep peak memory bounded for biobank-scale matrices.
    """
    n_markers, n_samples = lrr.shape

    Q, _ = np.linalg.qr(C)
    Y_resid = phenotype - Q @ (Q.T @ phenotype)

    beta = np.empty(n_markers)
    se = np.empty(n_markers)
    t_stat = np.empty(n_markers)
    p_value = np.empty(n_markers)
    n_per_marker = np.full(n_markers, n_samples, dtype=int)

    df = n_samples - df_cov - 1

    for start in range(0, n_markers, _OLS_CHUNK_SIZE):
        end = min(start + _OLS_CHUNK_SIZE, n_markers)
        chunk = lrr[start:end]

        X_resid = chunk - (chunk @ Q) @ Q.T

        xty = X_resid @ Y_resid
        xtx = np.sum(X_resid ** 2, axis=1)

        safe = xtx > 0
        b = np.where(safe, xty / np.where(safe, xtx, 1.0), 0.0)

        residuals = Y_resid[np.newaxis, :] - b[:, np.newaxis] * X_resid
        sigma2 = np.sum(residuals ** 2, axis=1) / max(df, 1)

        s = np.where(safe, np.sqrt(sigma2 / np.where(safe, xtx, 1.0)), np.inf)
        t = np.where(s < np.inf, b / np.where(s > 0, s, 1.0), 0.0)

        if df > 0:
            pv = 2.0 * stats.t.sf(np.abs(t), df=df)
        else:
            pv = np.ones(end - start)

        beta[start:end] = b
        se[start:end] = s
        t_stat[start:end] = t
        p_value[start:end] = pv

    return beta, se, t_stat, p_value, n_per_marker


def _ols_scan_missing(
    lrr: np.ndarray,
    phenotype: np.ndarray,
    C: np.ndarray,
    df_cov: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-marker OLS with complete-case analysis for missing LRR."""
    n_markers = lrr.shape[0]

    beta = np.zeros(n_markers)
    se = np.full(n_markers, np.inf)
    t_stat = np.zeros(n_markers)
    p_value = np.ones(n_markers)
    n_per_marker = np.zeros(n_markers, dtype=int)

    for i in range(n_markers):
        x = lrr[i]
        valid = ~np.isnan(x)
        n_valid = int(valid.sum())
        n_per_marker[i] = n_valid

        min_n = df_cov + 2
        if n_valid < min_n:
            continue

        x_i = x[valid]
        y_i = phenotype[valid]
        C_i = C[valid]

        Q, _ = np.linalg.qr(C_i)
        y_resid = y_i - Q @ (Q.T @ y_i)
        x_resid = x_i - Q @ (Q.T @ x_i)

        xtx = float(np.dot(x_resid, x_resid))
        if xtx <= 0:
            continue

        xty = float(np.dot(x_resid, y_resid))
        beta[i] = xty / xtx

        resid = y_resid - beta[i] * x_resid
        df = n_valid - df_cov - 1
        if df <= 0:
            continue

        sigma2 = float(np.sum(resid ** 2)) / df
        se[i] = np.sqrt(sigma2 / xtx)
        t_stat[i] = beta[i] / se[i]
        p_value[i] = 2.0 * stats.t.sf(abs(t_stat[i]), df=df)

    return beta, se, t_stat, p_value, n_per_marker


def _ols_scan(
    lrr: np.ndarray,
    phenotype: np.ndarray,
    covariates: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """OLS association scan: *phenotype ~ LRR_marker + covariates*.

    Parameters
    ----------
    lrr : ndarray, shape (n_markers, n_samples)
    phenotype : ndarray, shape (n_samples,)
    covariates : ndarray, shape (n_samples, n_covariates) or None

    Returns
    -------
    beta, se, t_stat, p_value, n_samples : arrays of shape (n_markers,)
    """
    n_samples = lrr.shape[1]

    if covariates is not None:
        C = np.column_stack([np.ones(n_samples), covariates])
    else:
        C = np.ones((n_samples, 1))

    df_cov = C.shape[1]

    if np.isnan(lrr).any():
        return _ols_scan_missing(lrr, phenotype, C, df_cov)
    return _ols_scan_complete(lrr, phenotype, C, df_cov)


# ---------------------------------------------------------------------------
# Logistic regression helpers
# ---------------------------------------------------------------------------

def _logistic_scan(
    lrr: np.ndarray,
    phenotype: np.ndarray,
    covariates: np.ndarray | None = None,
    max_iter: int = 25,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-marker logistic regression via IRLS.

    Parameters
    ----------
    lrr : ndarray, shape (n_markers, n_samples)
    phenotype : ndarray, shape (n_samples,) — binary (0/1)
    covariates : ndarray or None
    max_iter : int
    tol : float

    Returns
    -------
    beta, se, z_stat, p_value, n_samples : arrays of shape (n_markers,)
    """
    n_markers, n_samples = lrr.shape

    if covariates is not None:
        C = np.column_stack([np.ones(n_samples), covariates])
    else:
        C = np.ones((n_samples, 1))

    beta = np.zeros(n_markers)
    se = np.full(n_markers, np.inf)
    z_stat = np.zeros(n_markers)
    p_value = np.ones(n_markers)
    n_per_marker = np.zeros(n_markers, dtype=int)

    for i in range(n_markers):
        x = lrr[i]
        valid = ~np.isnan(x)
        n_valid = int(valid.sum())
        n_per_marker[i] = n_valid

        n_params = C.shape[1] + 1
        if n_valid < n_params + 1:
            continue

        x_i = x[valid]
        y_i = phenotype[valid]
        X = np.column_stack([C[valid], x_i])

        b = np.zeros(X.shape[1])
        converged = False
        for _ in range(max_iter):
            eta = X @ b
            eta = np.clip(eta, -_MAX_ETA, _MAX_ETA)
            mu = 1.0 / (1.0 + np.exp(-eta))
            w = mu * (1.0 - mu)
            w = np.maximum(w, 1e-10)

            z = eta + (y_i - mu) / w
            try:
                XtWX = X.T * w[np.newaxis, :] @ X
                XtWz = X.T @ (w * z)
                b_new = np.linalg.solve(XtWX, XtWz)
            except np.linalg.LinAlgError:
                break

            if np.max(np.abs(b_new - b)) < tol:
                b = b_new
                converged = True
                break
            b = b_new

        if not converged:
            continue

        try:
            eta = X @ b
            eta = np.clip(eta, -_MAX_ETA, _MAX_ETA)
            mu = 1.0 / (1.0 + np.exp(-eta))
            w = mu * (1.0 - mu)
            w = np.maximum(w, 1e-10)
            XtWX = X.T * w[np.newaxis, :] @ X
            XtWX_inv = np.linalg.inv(XtWX)
            beta[i] = b[-1]
            se[i] = np.sqrt(max(XtWX_inv[-1, -1], 0.0))
            if se[i] > 0:
                z_stat[i] = beta[i] / se[i]
                p_value[i] = 2.0 * stats.norm.sf(abs(z_stat[i]))
        except np.linalg.LinAlgError:
            continue

    return beta, se, z_stat, p_value, n_per_marker


# ---------------------------------------------------------------------------
# LMM helpers  (spectral decomposition approach)
# ---------------------------------------------------------------------------

def _reml_loglik(
    log10_delta: float,
    y_rot: np.ndarray,
    X_rot: np.ndarray,
    eigenvalues: np.ndarray,
) -> float:
    """Negative profile REML log-likelihood for the variance ratio.

    Parameters
    ----------
    log10_delta : float
        log10(δ) where δ = σ²_e / σ²_g.
    y_rot, X_rot : ndarray
        Phenotype and covariates rotated into GRM eigenbasis.
    eigenvalues : ndarray
        GRM eigenvalues.

    Returns
    -------
    neg_ll : float
        Negative REML log-likelihood (to be minimised).
    """
    delta = 10.0 ** log10_delta
    n = len(y_rot)
    p = X_rot.shape[1]

    D_inv = 1.0 / (eigenvalues + delta)  # weights

    # Weighted least squares for covariates under null
    XtDX = X_rot.T * D_inv[np.newaxis, :] @ X_rot
    XtDy = X_rot.T @ (D_inv * y_rot)
    try:
        beta_hat = np.linalg.solve(XtDX, XtDy)
    except np.linalg.LinAlgError:
        return 1e30

    resid = y_rot - X_rot @ beta_hat
    sigma2_g = float(np.sum(D_inv * resid ** 2)) / (n - p)
    if sigma2_g <= 0:
        return 1e30

    # REML log-likelihood (up to constant)
    ll = -0.5 * (
        np.sum(np.log(eigenvalues + delta))
        + (n - p) * np.log(sigma2_g)
        + np.linalg.slogdet(XtDX)[1]
    )
    return -ll  # negate for minimisation


def _estimate_delta(
    y_rot: np.ndarray,
    X_rot: np.ndarray,
    eigenvalues: np.ndarray,
) -> float:
    """Estimate δ = σ²_e / σ²_g via bounded REML optimisation."""
    result = minimize_scalar(
        _reml_loglik,
        bounds=(-5.0, 5.0),
        method="bounded",
        args=(y_rot, X_rot, eigenvalues),
        options={"xatol": 1e-4},
    )
    return 10.0 ** result.x


def _mean_impute_lrr(lrr: np.ndarray) -> np.ndarray:
    """Replace per-marker NaN values with the marker's mean.

    This allows the full-sample GRM eigenbasis to be reused for all
    markers, avoiding a costly O(N³) eigendecomposition per marker with
    missing data.  Mean-imputation is the standard approach in EMMA,
    SAIGE, and fastGWA.

    Parameters
    ----------
    lrr : ndarray, shape (n_markers, n_samples)

    Returns
    -------
    lrr_imputed : ndarray, shape (n_markers, n_samples)
        A copy of *lrr* with NaN values replaced by per-marker means.
    """
    lrr = lrr.copy()
    nan_mask = np.isnan(lrr)
    if not nan_mask.any():
        return lrr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        row_means = np.nanmean(lrr, axis=1)
    # Markers that are entirely NaN get mean = NaN; replace with 0.
    row_means = np.where(np.isnan(row_means), 0.0, row_means)
    inds = np.where(nan_mask)
    lrr[inds] = row_means[inds[0]]
    return lrr


def _lmm_scan(
    lrr: np.ndarray,
    phenotype: np.ndarray,
    grm: np.ndarray,
    covariates: np.ndarray | None = None,
    chunk_size: int = _LMM_CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """LMM association scan using spectral decomposition.

    Model: phenotype ~ LRR_marker + covariates + (1|GRM)

    The LRR rotation into the eigenbasis (``lrr @ U``) is performed in
    chunks of *chunk_size* markers at a time to keep memory bounded for
    biobank-scale matrices.

    Parameters
    ----------
    lrr : ndarray, shape (n_markers, n_samples)
    phenotype : ndarray, shape (n_samples,)
    grm : ndarray, shape (n_samples, n_samples)
    covariates : ndarray, shape (n_samples, n_covariates) or None
    chunk_size : int
        Number of markers to rotate at once (default: 10 000).

    Returns
    -------
    beta, se, t_stat, p_value, n_samples : arrays of shape (n_markers,)
    """
    n_markers, n_samples = lrr.shape

    # Build covariate matrix (with intercept)
    if covariates is not None:
        C = np.column_stack([np.ones(n_samples), covariates])
    else:
        C = np.ones((n_samples, 1))

    # Eigendecompose GRM
    eigenvalues, U = np.linalg.eigh(grm)
    # Clamp small eigenvalues to avoid numerical issues
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # Rotate phenotype and covariates (small: n_samples-sized)
    y_rot = U.T @ phenotype
    X_rot = U.T @ C

    # Estimate δ under null model (no marker effect)
    delta = _estimate_delta(y_rot, X_rot, eigenvalues)

    # Weights for WLS
    D_inv = 1.0 / (eigenvalues + delta)
    p = C.shape[1]

    has_nan = np.isnan(lrr).any()

    if has_nan:
        # Mean-impute missing LRR values per marker so the full-sample
        # GRM eigenbasis can be reused for all markers.  This avoids an
        # O(N³) eigendecomposition per marker with missing data and is
        # the standard approach in EMMA, SAIGE, and fastGWA.
        lrr = _mean_impute_lrr(lrr)

    # Complete-data path: rotate and scan in chunks
    beta_all = np.empty(n_markers)
    se_all = np.empty(n_markers)
    t_all = np.empty(n_markers)
    p_all = np.empty(n_markers)
    n_all = np.empty(n_markers, dtype=int)

    for start in range(0, n_markers, chunk_size):
        end = min(start + chunk_size, n_markers)
        lrr_rot_chunk = lrr[start:end] @ U  # shape (chunk, n_samples)
        b, s, t, pv, ns = _lmm_scan_complete(
            lrr_rot_chunk, y_rot, X_rot, eigenvalues, delta, D_inv,
            n_samples, p,
        )
        beta_all[start:end] = b
        se_all[start:end] = s
        t_all[start:end] = t
        p_all[start:end] = pv
        n_all[start:end] = ns

    return beta_all, se_all, t_all, p_all, n_all


def _lmm_scan_complete(
    lrr_rot: np.ndarray,
    y_rot: np.ndarray,
    X_rot: np.ndarray,
    eigenvalues: np.ndarray,
    delta: float,
    D_inv: np.ndarray,
    n_samples: int,
    p: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised LMM scan when no LRR values are missing."""
    n_markers = lrr_rot.shape[0]

    # Weight-project covariates out of y and LRR in the rotated space
    # Weighted QR: scale by sqrt(D_inv)
    sqrt_w = np.sqrt(D_inv)
    X_w = X_rot * sqrt_w[:, np.newaxis]
    y_w = y_rot * sqrt_w
    lrr_w = lrr_rot * sqrt_w[np.newaxis, :]

    Q, _ = np.linalg.qr(X_w)
    y_resid = y_w - Q @ (Q.T @ y_w)
    z_resid = lrr_w - (lrr_w @ Q) @ Q.T

    # Univariate WLS
    ztDz = np.sum(z_resid ** 2, axis=1)
    ztDy = z_resid @ y_resid

    safe = ztDz > 0
    beta = np.where(safe, ztDy / np.where(safe, ztDz, 1.0), 0.0)

    residuals = y_resid[np.newaxis, :] - beta[:, np.newaxis] * z_resid
    df = n_samples - p - 1
    sigma2 = np.sum(residuals ** 2, axis=1) / max(df, 1)

    se = np.where(safe, np.sqrt(sigma2 / np.where(safe, ztDz, 1.0)), np.inf)
    t_stat = np.where(se < np.inf, beta / np.where(se > 0, se, 1.0), 0.0)

    if df > 0:
        p_value = 2.0 * stats.t.sf(np.abs(t_stat), df=df)
    else:
        p_value = np.ones(n_markers)

    n_per_marker = np.full(n_markers, n_samples, dtype=int)
    return beta, se, t_stat, p_value, n_per_marker



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_association(
    lrr: np.ndarray,
    phenotype: np.ndarray,
    variants: list[dict[str, object]],
    covariates: np.ndarray | None = None,
    method: str = "lmm",
    grm: np.ndarray | None = None,
) -> AssociationResult:
    """Run genome-wide association of phenotype on per-marker LRR.

    For each marker *i* the model is::

        phenotype ~ LRR_i + covariates                  (OLS)
        phenotype ~ LRR_i + covariates + (1|GRM)        (LMM)
        logit(P(phenotype=1)) ~ LRR_i + covariates      (logistic)

    Parameters
    ----------
    lrr : ndarray, shape (n_markers, n_samples)
        Batch-corrected LRR matrix (rows = markers, columns = samples).
    phenotype : ndarray, shape (n_samples,)
        Phenotype vector.  Continuous for *method='ols'* or *'lmm'*,
        binary (0/1) for *method='logistic'*.
    variants : list of dict
        Per-marker metadata.  Each dict must contain ``'chrom'`` and
        ``'pos'``; ``'id'`` is used if present, otherwise
        ``'chrom:pos'`` is constructed.
    covariates : ndarray, shape (n_samples, n_covariates), optional
        Covariate matrix (**without** an intercept; one is added
        automatically).  For LMM, these should include genetic PCs.
    method : ``'lmm'`` | ``'ols'`` | ``'logistic'``
        Association method.  ``'lmm'`` (default) requires *grm*.
    grm : ndarray, shape (n_samples, n_samples), optional
        Genetic Relationship Matrix.  Required when *method='lmm'*.

    Returns
    -------
    AssociationResult

    Raises
    ------
    ValueError
        If dimensions are inconsistent, *method* is unknown, or
        *grm* is missing for LMM.
    """
    n_markers, n_samples = lrr.shape
    if phenotype.ndim != 1 or phenotype.shape[0] != n_samples:
        raise ValueError(
            f"phenotype must be 1-D with {n_samples} elements, "
            f"got shape {phenotype.shape}"
        )
    if len(variants) != n_markers:
        raise ValueError(
            f"len(variants) ({len(variants)}) != n_markers ({n_markers})"
        )
    if covariates is not None:
        if covariates.ndim != 2 or covariates.shape[0] != n_samples:
            raise ValueError(
                f"covariates must have shape ({n_samples}, k), "
                f"got {covariates.shape}"
            )

    if method == "lmm":
        if grm is None:
            raise ValueError("LMM method requires a GRM (grm=...)")
        if grm.shape != (n_samples, n_samples):
            raise ValueError(
                f"GRM must have shape ({n_samples}, {n_samples}), "
                f"got {grm.shape}"
            )
        y_valid = phenotype[~np.isnan(phenotype)]
        unique = np.unique(y_valid)
        if unique.size > 0 and np.all(np.isin(unique, np.array([0.0, 1.0]))):
            case_fraction = float(np.mean(y_valid))
            if (
                case_fraction < _LMM_BINARY_CASE_FRAC_WARN_MIN
                or case_fraction > _LMM_BINARY_CASE_FRAC_WARN_MAX
            ):
                n_cases = int(round(case_fraction * y_valid.size))
                logger.warning(
                    "LMM selected with a binary phenotype and an unbalanced "
                    "case fraction (%.3f; %d cases / %d samples). Treating "
                    "a highly unbalanced binary trait as continuous can "
                    "inflate Type I error, especially for lower-frequency "
                    "variants. Consider (a) pre-filtering related "
                    "individuals and using --method logistic, or (b) "
                    "focusing interpretation on variants where "
                    "(MAF * number_of_cases) > 100.",
                    case_fraction, n_cases, y_valid.size,
                )
        beta, se, stat, pval, ns = _lmm_scan(
            lrr, phenotype, grm, covariates,
        )
    elif method == "ols":
        beta, se, stat, pval, ns = _ols_scan(lrr, phenotype, covariates)
    elif method == "logistic":
        unique = np.unique(phenotype[~np.isnan(phenotype)])
        if not np.array_equal(unique, np.array([0.0, 1.0])):
            raise ValueError(
                "Logistic regression requires binary phenotype (0/1); "
                f"got unique values {unique}"
            )
        if grm is not None:
            logger.warning(
                "Logistic regression does not use the GRM for relatedness "
                "correction. Only fixed-effect covariates (e.g. PCs) are "
                "applied. Consider pre-filtering highly related individuals "
                "or using --method lmm with a continuous phenotype proxy."
            )
        beta, se, stat, pval, ns = _logistic_scan(lrr, phenotype, covariates)
    else:
        raise ValueError(
            f"Unknown method {method!r}; use 'lmm', 'ols', or 'logistic'"
        )

    chroms = [v["chrom"] for v in variants]
    positions = [v["pos"] for v in variants]
    ids = [
        v.get("id") or f"{v['chrom']}:{v['pos']}" for v in variants
    ]

    return AssociationResult(
        chrom=chroms,
        pos=positions,
        variant_id=ids,
        beta=beta,
        se=se,
        stat=stat,
        p_value=pval,
        n_samples=ns,
        method=method,
    )


def run_association_streaming(
    lrr_chunks,
    phenotype: np.ndarray,
    covariates: np.ndarray | None = None,
    method: str = "lmm",
    grm: np.ndarray | None = None,
    *,
    exclude_monomorphic: bool = True,
    exclude_intensity_only: bool = True,
    n_markers_total: int | None = None,
) -> tuple["AssociationResult", dict]:
    """Stream-based association scan that never holds the full LRR matrix.

    Instead of accepting a pre-loaded ``lrr`` matrix, this function
    consumes an iterable of ``(lrr_chunk, variants_chunk)`` tuples —
    typically produced by :func:`array_lrr_gwas.io_vcf.stream_lrr_chunks`.

    Per-chunk marker filtering (INTENSITY_ONLY and monomorphic LRR) is
    applied inline, so the caller does not need to pre-filter.

    Parameters
    ----------
    lrr_chunks : iterable of (ndarray, list[dict])
        Each element is ``(lrr_chunk, variants_chunk)`` where
        ``lrr_chunk`` has shape ``(k, n_samples)`` and
        ``variants_chunk`` is a list of *k* variant metadata dicts.
    phenotype : ndarray, shape (n_samples,)
        Phenotype vector.
    covariates : ndarray, shape (n_samples, n_covariates), optional
        Covariate matrix (without intercept).
    method : ``'lmm'`` | ``'ols'`` | ``'logistic'``
        Association method.
    grm : ndarray, shape (n_samples, n_samples), optional
        GRM required for method ``'lmm'``.
    exclude_monomorphic : bool
        Whether to skip markers with zero LRR variance (default True).
    exclude_intensity_only : bool
        Whether to skip INTENSITY_ONLY markers (default True).
    n_markers_total : int, optional
        Expected total number of markers (used to set the progress bar
        total).  When ``None`` the bar runs in indeterminate mode.

    Returns
    -------
    result : AssociationResult
    exclusion_info : dict
        Keys: ``n_total``, ``n_tested``, ``n_intensity_only``,
        ``n_monomorphic``, ``excluded_markers`` (dict mapping ID→reason),
        ``tested_mono_flags`` (list[bool] for surviving markers).
    """
    n_samples = phenotype.shape[0]

    # --- Pre-compute LMM invariants (eigenbasis, delta) ---
    lmm_invariants = None
    if method == "lmm":
        if grm is None:
            raise ValueError("LMM method requires a GRM (grm=...)")
        if grm.shape != (n_samples, n_samples):
            raise ValueError(
                f"GRM must have shape ({n_samples}, {n_samples}), "
                f"got {grm.shape}"
            )
        if covariates is not None:
            C = np.column_stack([np.ones(n_samples), covariates])
        else:
            C = np.ones((n_samples, 1))

        eigenvalues, U = np.linalg.eigh(grm)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        y_rot = U.T @ phenotype
        X_rot = U.T @ C
        delta = _estimate_delta(y_rot, X_rot, eigenvalues)
        D_inv = 1.0 / (eigenvalues + delta)
        lmm_invariants = (U, eigenvalues, y_rot, X_rot, delta, D_inv, C.shape[1])

    # Accumulators
    chroms: list[str] = []
    positions: list[int] = []
    ids: list[str] = []
    betas: list[np.ndarray] = []
    ses: list[np.ndarray] = []
    stats_: list[np.ndarray] = []
    pvals: list[np.ndarray] = []
    ns: list[np.ndarray] = []
    tested_mono: list[bool] = []

    n_total = 0
    n_intensity = 0
    n_mono = 0
    excluded_markers: dict[str, str] = {}

    scan_pbar = tqdm(
        total=n_markers_total,
        desc=f"{method} scan",
        unit="marker",
        leave=False,
        dynamic_ncols=True,
    )
    try:
        for lrr_chunk, variants_chunk in lrr_chunks:
            n_chunk = lrr_chunk.shape[0]
            n_total += n_chunk

            # Per-chunk marker exclusion
            keep = np.ones(n_chunk, dtype=bool)

            if exclude_intensity_only:
                io_mask = np.array(
                    [v.get("intensity_only", False) for v in variants_chunk],
                    dtype=bool,
                )
                n_io = int(io_mask.sum())
                n_intensity += n_io
                keep &= ~io_mask

            # Monomorphic check: a marker is monomorphic when nanmin == nanmax
            # (all finite values identical) or the entire row is NaN.
            # Using nanmin/nanmax is more robust to floating-point imprecision
            # than exact equality on nanvar (which can yield 1e-16 for identical
            # floats after arithmetic operations).
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                _lrr_min = np.nanmin(lrr_chunk, axis=1)
                _lrr_max = np.nanmax(lrr_chunk, axis=1)
            all_nan = np.all(np.isnan(lrr_chunk), axis=1)
            mono_mask = (_lrr_min == _lrr_max) | all_nan

            if exclude_monomorphic:
                n_m = int((mono_mask & keep).sum())
                n_mono += n_m
                keep &= ~mono_mask

            # Record exclusions
            for i, (v, k) in enumerate(zip(variants_chunk, keep)):
                vid = v.get("id") or f"{v['chrom']}:{v['pos']}"
                if not k:
                    reasons = []
                    if exclude_intensity_only and v.get("intensity_only", False):
                        reasons.append("intensity_only")
                    if exclude_monomorphic and mono_mask[i]:
                        reasons.append("monomorphic_lrr")
                    excluded_markers[vid] = ";".join(reasons) if reasons else "excluded"

            n_keep = int(keep.sum())
            scan_pbar.update(n_chunk)
            scan_pbar.set_postfix(
                tested=len(chroms) + n_keep,
                mono_excl=n_mono,
                refresh=False,
            )
            if n_keep == 0:
                continue

            lrr_keep = lrr_chunk[keep]
            vars_keep = [v for v, k in zip(variants_chunk, keep) if k]
            mono_keep = mono_mask[keep]

            # Track monomorphic flags for surviving markers
            tested_mono.extend(bool(m) for m in mono_keep)

            # Run scan on this chunk
            if method == "lmm":
                U, eigenvalues, y_rot, X_rot, delta, D_inv, p = lmm_invariants
                lrr_imp = _mean_impute_lrr(lrr_keep) if np.isnan(lrr_keep).any() else lrr_keep
                lrr_rot = lrr_imp @ U
                b, s, t, pv, n = _lmm_scan_complete(
                    lrr_rot, y_rot, X_rot, eigenvalues, delta, D_inv,
                    n_samples, p,
                )
            elif method == "ols":
                b, s, t, pv, n = _ols_scan(lrr_keep, phenotype, covariates)
            elif method == "logistic":
                b, s, t, pv, n = _logistic_scan(lrr_keep, phenotype, covariates)
            else:
                raise ValueError(f"Unknown method {method!r}")

            for v in vars_keep:
                chroms.append(v["chrom"])
                positions.append(v["pos"])
                ids.append(v.get("id") or f"{v['chrom']}:{v['pos']}")

            betas.append(b)
            ses.append(s)
            stats_.append(t)
            pvals.append(pv)
            ns.append(n)
    finally:
        scan_pbar.close()

    # Concatenate results
    if betas:
        beta_arr = np.concatenate(betas)
        se_arr = np.concatenate(ses)
        stat_arr = np.concatenate(stats_)
        pval_arr = np.concatenate(pvals)
        n_arr = np.concatenate(ns)
    else:
        beta_arr = np.empty(0)
        se_arr = np.empty(0)
        stat_arr = np.empty(0)
        pval_arr = np.empty(0)
        n_arr = np.empty(0, dtype=int)

    result = AssociationResult(
        chrom=chroms,
        pos=positions,
        variant_id=ids,
        beta=beta_arr,
        se=se_arr,
        stat=stat_arr,
        p_value=pval_arr,
        n_samples=n_arr,
        method=method,
    )

    exclusion_info = {
        "n_total": n_total,
        "n_tested": len(chroms),
        "n_intensity_only": n_intensity,
        "n_monomorphic": n_mono,
        "excluded_markers": excluded_markers,
        "tested_mono_flags": tested_mono,
    }

    return result, exclusion_info
