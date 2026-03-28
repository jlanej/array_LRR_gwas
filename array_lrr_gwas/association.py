"""GWAS Association Engine for Continuous LRR Predictors.

Engine Evaluation
-----------------
The following engines were evaluated for compatibility with using LRR
as a continuous predictor in genome-wide association:

**PLINK2**
  Designed for discrete genotype dosages (0/1/2).  Can accept dosage
  inputs, but continuous LRR values fall outside the expected [0, 2]
  range and may trigger warnings or internal clipping.  Linear/logistic
  models assume genotype coding; residualisation and standard-error
  estimation may behave unexpectedly with unbounded continuous
  predictors.
  *Verdict:* Not directly suitable without non-standard workarounds.

**REGENIE**
  Two-step method (ridge regression followed by single-variant tests)
  that explicitly assumes biallelic genotype coding (0/1/2).  Step 1
  whole-genome model relies on LD structure and genotype assumptions
  that do not apply to LRR.
  *Verdict:* Incompatible with continuous LRR input.

**TensorQTL**
  Designed for cis/trans-eQTL mapping (expression ~ genotype).  Uses
  GPU-accelerated OLS; could in principle be repurposed by swapping the
  genotype matrix for LRR.  Adds a heavy PyTorch dependency and
  requires GPU for performance.
  *Verdict:* Technically adaptable but impractical dependency burden.

**Hail**
  Spark/JVM-based; ``linear_regression_rows()`` accepts continuous
  predictors in principle.  Extremely heavy infrastructure requirement
  for what reduces to OLS.
  *Verdict:* Overkill; not justified for this use case.

Chosen Approach: Pure NumPy/SciPy OLS
--------------------------------------
* Frisch-Waugh-Lovell theorem enables vectorised covariate adjustment.
* Handles missing data per-marker via complete-case masking.
* Logistic regression via iteratively reweighted least squares (IRLS)
  for binary phenotypes.
* No external binary dependencies beyond NumPy and SciPy.
* Portable, testable, and ready for containerised deployment.

Limitations and Best Practices
------------------------------
* Population structure and relatedness are **not** modelled internally.
  For large biobank-scale studies, pre-compute principal components or
  a GRM externally and supply them as covariates.
* Per-marker missing-data handling falls back to a slower loop; ensure
  upstream QC minimises missingness for best performance.
* Logistic regression uses IRLS with a fixed iteration cap; convergence
  failures are silently skipped (NaN beta / p = 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class AssociationResult:
    """Per-marker association summary statistics."""

    chrom: List[str]
    pos: List[int]
    variant_id: List[str]
    beta: np.ndarray
    se: np.ndarray
    stat: np.ndarray
    p_value: np.ndarray
    n_samples: np.ndarray
    method: str

    def to_records(self) -> List[Dict[str, Any]]:
        """Convert to a list of dicts (one per variant)."""
        records: List[Dict[str, Any]] = []
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
    covariates: Optional[np.ndarray] = None,
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
    covariates: Optional[np.ndarray] = None,
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
            eta = np.clip(eta, -20, 20)
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
            eta = np.clip(eta, -20, 20)
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
# Public API
# ---------------------------------------------------------------------------

def run_association(
    lrr: np.ndarray,
    phenotype: np.ndarray,
    variants: List[Dict[str, Any]],
    covariates: Optional[np.ndarray] = None,
    method: str = "ols",
) -> AssociationResult:
    """Run genome-wide association of phenotype on per-marker LRR.

    For each marker *i* the model is::

        phenotype ~ LRR_i + covariates   (OLS or logistic)

    Parameters
    ----------
    lrr : ndarray, shape (n_markers, n_samples)
        Batch-corrected LRR matrix (rows = markers, columns = samples).
    phenotype : ndarray, shape (n_samples,)
        Phenotype vector.  Continuous for *method='ols'*, binary (0/1)
        for *method='logistic'*.
    variants : list of dict
        Per-marker metadata.  Each dict must contain ``'chrom'`` and
        ``'pos'``; ``'id'`` is used if present, otherwise
        ``'chrom:pos'`` is constructed.
    covariates : ndarray, shape (n_samples, n_covariates), optional
        Covariate matrix (**without** an intercept; one is added
        automatically).
    method : ``'ols'`` | ``'logistic'``

    Returns
    -------
    AssociationResult

    Raises
    ------
    ValueError
        If dimensions are inconsistent or *method* is unknown.
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

    if method == "ols":
        beta, se, stat, pval, ns = _ols_scan(lrr, phenotype, covariates)
    elif method == "logistic":
        unique = np.unique(phenotype[~np.isnan(phenotype)])
        if not np.array_equal(unique, np.array([0.0, 1.0])):
            raise ValueError(
                "Logistic regression requires binary phenotype (0/1); "
                f"got unique values {unique}"
            )
        beta, se, stat, pval, ns = _logistic_scan(lrr, phenotype, covariates)
    else:
        raise ValueError(f"Unknown method {method!r}; use 'ols' or 'logistic'")

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
