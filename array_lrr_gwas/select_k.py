"""Automated selection of the number of components *k*.

Two complementary heuristics are provided:

1. **Marchenko–Pastur (MP) threshold** – retains only singular values that
   exceed the upper edge of the Marchenko–Pastur distribution expected
   under a pure-noise null.  This is grounded in random-matrix theory and
   provides a principled, data-driven cutoff.

2. **Elbow detection** – locates the "knee" in the scree plot of singular
   values by finding the point of maximum second-order difference.  This
   is a widely used heuristic when the noise floor is not well-characterised.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _mp_upper_edge(n_markers: int, n_samples: int, sigma2: float) -> float:
    """Upper edge of the Marchenko–Pastur distribution.

    Parameters
    ----------
    n_markers, n_samples : int
        Dimensions of the centred data matrix.
    sigma2 : float
        Estimated noise variance (median eigenvalue is a robust estimator).

    Returns
    -------
    lambda_plus : float
        Upper edge: eigenvalues above this are unlikely under pure noise.
    """
    gamma = n_markers / n_samples
    return sigma2 * (1 + np.sqrt(gamma)) ** 2


def select_k_mp(
    singular_values: NDArray[np.floating],
    n_markers: int,
    n_samples: int,
) -> int:
    """Select *k* using the Marchenko–Pastur threshold.

    The noise variance is estimated as the median squared singular value
    divided by the Marchenko–Pastur median (a robust choice when signal
    components are relatively few).

    Parameters
    ----------
    singular_values : 1-D array
        Singular values in descending order (from a truncated SVD on the
        centred markers × samples matrix).
    n_markers, n_samples : int
        Original matrix dimensions *before* truncation.

    Returns
    -------
    k : int
        Recommended number of components (at least 1).
    """
    s2 = singular_values ** 2

    # Robust noise estimate: use the smallest singular value squared,
    # normalised to an eigenvalue of the covariance matrix.
    noise_eigenvalue = s2[-1] / n_samples
    upper = _mp_upper_edge(n_markers, n_samples, noise_eigenvalue)

    # Eigenvalues of the covariance matrix = s^2 / n_samples
    eigenvalues = s2 / n_samples
    k = int(np.sum(eigenvalues > upper))
    return max(k, 1)


def select_k_elbow(
    singular_values: NDArray[np.floating],
    max_k: int | None = None,
) -> int:
    """Select *k* via the elbow (knee) of the scree plot.

    The elbow is detected as the index of the largest second-order
    finite difference of the singular values.

    Parameters
    ----------
    singular_values : 1-D array
        Singular values in descending order.
    max_k : int or None
        Upper bound on the returned *k*.

    Returns
    -------
    k : int
        Recommended number of components (at least 1).
    """
    s = np.asarray(singular_values, dtype=np.float64)
    if len(s) < 3:
        return 1
    # Second-order finite difference: large positive values indicate a
    # transition from rapid to slow decay (the "elbow").
    d2 = np.diff(s, n=2)
    # The elbow is at the position where the curvature is maximal.
    # diff reduces length by 2, so index i in d2 corresponds to
    # component i+1 in the original series.
    elbow = int(np.argmax(d2)) + 1
    k = max(elbow, 1)
    if max_k is not None:
        k = min(k, max_k)
    return k
