"""GWAS association engine for continuous LRR predictors.

Provides LMM (default), OLS, and logistic regression association scans.
The LMM uses spectral decomposition of a GRM with profile-REML variance
component estimation (FaST-LMM / EMMA approach).

See ``docs/association_engine_design.md`` for the full engine evaluation
and design rationale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)


# Logistic IRLS: clamp linear predictor to avoid overflow in exp().
_MAX_ETA: float = 20.0

# LMM: number of markers to rotate into the eigenbasis at once.
# Keeps peak memory bounded for biobank-scale LRR matrices.
_LMM_CHUNK_SIZE: int = 10_000

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
    """
    n_markers, n_samples = lrr.shape

    Q, _ = np.linalg.qr(C)
    Y_resid = phenotype - Q @ (Q.T @ phenotype)
    X_resid = lrr - (lrr @ Q) @ Q.T

    xty = X_resid @ Y_resid
    xtx = np.sum(X_resid ** 2, axis=1)

    safe = xtx > 0
    beta = np.where(safe, xty / np.where(safe, xtx, 1.0), 0.0)

    residuals = Y_resid[np.newaxis, :] - beta[:, np.newaxis] * X_resid
    df = n_samples - df_cov - 1
    sigma2 = np.sum(residuals ** 2, axis=1) / max(df, 1)

    se = np.where(safe, np.sqrt(sigma2 / np.where(safe, xtx, 1.0)), np.inf)
    t_stat = np.where(se < np.inf, beta / np.where(se > 0, se, 1.0), 0.0)

    if df > 0:
        p_value = 2.0 * stats.t.sf(np.abs(t_stat), df=df)
    else:
        p_value = np.ones(n_markers)

    n_per_marker = np.full(n_markers, n_samples, dtype=int)
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
        # Missing-data path doesn't benefit from chunking (per-marker loop)
        return _lmm_scan_missing(
            lrr, phenotype, C, grm, eigenvalues, U, delta,
        )

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


def _lmm_scan_missing(
    lrr: np.ndarray,
    phenotype: np.ndarray,
    C: np.ndarray,
    grm: np.ndarray,
    eigenvalues: np.ndarray,
    U: np.ndarray,
    delta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-marker LMM with complete-case analysis for missing LRR."""
    n_markers, n_samples = lrr.shape
    p = C.shape[1]

    D_inv_full = 1.0 / (eigenvalues + delta)

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

        min_n = p + 2
        if n_valid < min_n:
            continue

        if n_valid == n_samples:
            # Use full rotated space
            x_rot = x @ U
            sqrt_w = np.sqrt(D_inv_full)
            X_w = (U.T @ C) * sqrt_w[:, np.newaxis]
            y_w = (U.T @ phenotype) * sqrt_w
            z_w = x_rot * sqrt_w
        else:
            # Subset and re-decompose for this marker
            x_i = x[valid]
            y_i = phenotype[valid]
            C_i = C[valid]
            grm_i = grm[np.ix_(valid, valid)]

            evals_i, U_i = np.linalg.eigh(grm_i)
            evals_i = np.maximum(evals_i, 0.0)
            D_inv_i = 1.0 / (evals_i + delta)

            sqrt_w = np.sqrt(D_inv_i)
            X_w = (U_i.T @ C_i) * sqrt_w[:, np.newaxis]
            y_w = (U_i.T @ y_i) * sqrt_w
            z_w = (U_i.T @ x_i) * sqrt_w

        Q, _ = np.linalg.qr(X_w)
        y_r = y_w - Q @ (Q.T @ y_w)
        z_r = z_w - Q @ (Q.T @ z_w)

        ztDz = float(np.dot(z_r, z_r))
        if ztDz <= 0:
            continue

        ztDy = float(np.dot(z_r, y_r))
        beta[i] = ztDy / ztDz

        resid = y_r - beta[i] * z_r
        df = n_valid - p - 1
        if df <= 0:
            continue

        sigma2 = float(np.sum(resid ** 2)) / df
        se[i] = np.sqrt(sigma2 / ztDz)
        t_stat[i] = beta[i] / se[i]
        p_value[i] = 2.0 * stats.t.sf(abs(t_stat[i]), df=df)

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
