"""Decomposition routines for batch-effect estimation.

The default backend is :func:`sklearn.utils.extmath.randomized_svd`, which
provides a fast randomized SVD suitable for large genotyping arrays.
An *fbpca* backend is available when the optional ``fbpca`` package is
installed.  Users may also supply any callable with the signature
``(matrix, k) -> (U, S, Vt)`` to plug in alternative algorithms.
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np
from numpy.typing import NDArray
from sklearn.utils.extmath import randomized_svd as _sklearn_rsvd


class DecompCallable(Protocol):
    """Protocol for a pluggable decomposition function."""

    def __call__(
        self,
        matrix: NDArray[np.floating],
        k: int,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Return (U, S, Vt) truncated to *k* components."""
        ...


def rsvd(
    matrix: NDArray[np.floating],
    k: int,
    *,
    n_oversamples: int = 10,
    n_iter: int = 5,
    random_state: int | np.random.RandomState | None = 0,
) -> tuple[NDArray, NDArray, NDArray]:
    """Randomized SVD via scikit-learn.

    Parameters
    ----------
    matrix : ndarray, shape (m, n)
        Input matrix (e.g. centred LRR values, markers × samples).
    k : int
        Number of singular-value components to compute.
    n_oversamples : int
        Additional columns for the random projection (improves accuracy).
    n_iter : int
        Number of power iterations for improved approximation.
    random_state : int, RandomState, or None
        Seed for reproducibility.

    Returns
    -------
    U : ndarray, shape (m, k)
    s : ndarray, shape (k,)
    Vt : ndarray, shape (k, n)
    """
    return _sklearn_rsvd(
        matrix,
        n_components=k,
        n_oversamples=n_oversamples,
        n_iter=n_iter,
        random_state=random_state,
    )


def _fbpca_backend(
    matrix: NDArray[np.floating],
    k: int,
) -> tuple[NDArray, NDArray, NDArray]:
    """Decomposition using Facebook's fbpca library.

    Raises
    ------
    ImportError
        If ``fbpca`` is not installed.
    """
    try:
        import fbpca  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "fbpca is not installed. Install it with: pip install fbpca"
        ) from exc
    U, s, Vt = fbpca.pca(matrix, k=k, raw=True)
    return U, s, Vt


def estimate_rsvd_marker_budget(
    n_hq_samples: int,
    n_components: int,
    *,
    max_ram_bytes: int,
    bytes_per_element: int = 8,
    safety_factor: float = 2.5,
) -> int:
    """Return the maximum number of markers that fit within *max_ram_bytes*.

    The dominant memory cost of the RSVD decomposition is the input matrix
    copy plus internal working matrices, which scale as
    ``safety_factor × n_markers × n_hq_samples × bytes_per_element``.

    Parameters
    ----------
    n_hq_samples : int
        Number of high-quality samples (columns of the RSVD input).
    n_components : int
        Number of components *k* (used to refine the estimate).
    max_ram_bytes : int
        Maximum RAM budget in bytes.
    bytes_per_element : int
        Bytes per matrix element (8 for float64).
    safety_factor : float
        Multiplier applied to the theoretical minimum to account for
        Python overhead, sklearn temporaries, and OS buffers.
    """
    if max_ram_bytes <= 0:
        return 1
    denominator = safety_factor * n_hq_samples * bytes_per_element
    if denominator <= 0:
        return 1
    budget = int(max_ram_bytes / denominator)
    return max(1, budget)


def decompose(
    lrr: NDArray[np.floating],
    k: int,
    *,
    backend: str | DecompCallable = "rsvd",
) -> tuple[NDArray, NDArray, NDArray]:
    """Run a truncated decomposition on a centred LRR matrix.

    The input matrix is mean-centred per marker (row) before
    decomposition.  Missing values (``NaN``) are replaced with zero
    (i.e. the row mean) prior to decomposition; this is a standard
    imputation strategy for PCA on genotype-intensity data.

    Parameters
    ----------
    lrr : ndarray, shape (n_markers, n_samples)
        Raw or already-centred LRR values.
    k : int
        Number of components to extract.
    backend : ``"rsvd"`` | ``"fbpca"`` | callable(matrix, k) -> (U, s, Vt)
        Decomposition algorithm.

    Returns
    -------
    U : ndarray, shape (n_markers, k)
        Left singular vectors (marker loadings).
    s : ndarray, shape (k,)
        Singular values.
    Vt : ndarray, shape (k, n_samples)
        Right singular vectors (sample scores, transposed).
    """
    if lrr.ndim != 2:
        raise ValueError("lrr must be a 2-D array (markers × samples)")
    if k < 1:
        raise ValueError("k must be >= 1")
    if k > min(lrr.shape):
        raise ValueError(
            f"k={k} exceeds matrix dimensions {lrr.shape}"
        )

    mat = lrr.copy().astype(np.float64)
    row_means = np.nanmean(mat, axis=1, keepdims=True)
    mat -= row_means
    mat[np.isnan(mat)] = 0.0

    if callable(backend):
        func: DecompCallable = backend
    elif backend == "rsvd":

        def _rsvd_wrapper(m: NDArray, k_: int) -> tuple[NDArray, NDArray, NDArray]:
            return rsvd(m, k_)

        func = _rsvd_wrapper
    elif backend == "fbpca":
        func = _fbpca_backend
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    return func(mat, k)
