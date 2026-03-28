"""Genetic Relationship Matrix (GRM) computation.

Computes a sample-by-sample GRM from an additive dosage matrix using
the standard allele-frequency-standardised estimator (Yang *et al.*
2011, *Nat Genet* 43:519–525).

The GRM is defined as:

.. math::

    \\hat{K} = \\frac{1}{M} Z Z^\\top

where *Z* is the *n × M* matrix of standardised genotypes:

.. math::

    z_{ij} = \\frac{x_{ij} - 2 p_j}{\\sqrt{2 p_j (1 - p_j)}}

Missing genotypes are mean-imputed *before* standardisation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_grm(
    dosage: NDArray[np.floating],
    *,
    min_maf: float = 0.01,
) -> NDArray[np.floating]:
    """Compute the genomic relationship matrix.

    Parameters
    ----------
    dosage : ndarray, shape (n_variants, n_samples)
        Additive dosage matrix (0/1/2 with possible ``np.nan``).
    min_maf : float
        Variants with MAF below this threshold are excluded from the
        GRM calculation.

    Returns
    -------
    grm : ndarray, shape (n_samples, n_samples)
        Symmetric, positive-semidefinite GRM.

    Raises
    ------
    ValueError
        If no variants remain after MAF filtering.
    """
    n_variants, n_samples = dosage.shape

    # Mean-impute missing values per variant
    Z = dosage.copy()
    for i in range(n_variants):
        row = Z[i]
        mask = np.isnan(row)
        if mask.any():
            row[mask] = np.nanmean(row)

    # Compute allele frequencies
    freq = Z.mean(axis=1) / 2.0
    maf = np.minimum(freq, 1.0 - freq)

    # MAF filter
    keep = maf >= min_maf
    Z = Z[keep]
    freq = freq[keep]

    if Z.shape[0] == 0:
        raise ValueError(
            "No variants remain after MAF filtering "
            f"(min_maf={min_maf}); cannot compute GRM."
        )

    m = Z.shape[0]

    # Standardise: z_ij = (x_ij - 2p_j) / sqrt(2 p_j (1 - p_j))
    denom = np.sqrt(2.0 * freq * (1.0 - freq))
    safe = denom > 0
    Z[safe] = (Z[safe] - 2.0 * freq[safe, np.newaxis]) / denom[safe, np.newaxis]
    Z[~safe] = 0.0  # monomorphic after imputation

    # GRM = (1/M) Z^T Z  (Z is m×n, we want n×n)
    grm = (Z.T @ Z) / m
    return grm
