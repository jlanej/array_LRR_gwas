"""Tests for decomposition module."""

import numpy as np
import pytest

from array_lrr_gwas.decomposition import decompose, rsvd


class TestRsvd:
    def test_shapes(self):
        rng = np.random.default_rng(0)
        mat = rng.normal(0, 1, (50, 20))
        U, s, Vt = rsvd(mat, k=3)
        assert U.shape == (50, 3)
        assert s.shape == (3,)
        assert Vt.shape == (3, 20)

    def test_singular_values_descending(self):
        rng = np.random.default_rng(0)
        mat = rng.normal(0, 1, (50, 20))
        _, s, _ = rsvd(mat, k=5)
        assert np.all(np.diff(s) <= 0), "Singular values should be descending"

    def test_reproducible(self):
        rng = np.random.default_rng(0)
        mat = rng.normal(0, 1, (30, 10))
        _, s1, _ = rsvd(mat, k=3, random_state=42)
        _, s2, _ = rsvd(mat, k=3, random_state=42)
        np.testing.assert_array_equal(s1, s2)


class TestDecompose:
    def test_basic_output(self, synthetic_lrr):
        # Use only valid rows (no all-NaN rows)
        U, s, Vt = decompose(synthetic_lrr, k=3)
        assert U.shape == (synthetic_lrr.shape[0], 3)
        assert s.shape == (3,)
        assert Vt.shape == (3, synthetic_lrr.shape[1])

    def test_nan_handling(self):
        """NaN values should be imputed (to row mean = 0 after centring)."""
        mat = np.ones((10, 5))
        mat[0, 0] = np.nan
        U, s, Vt = decompose(mat, k=1)
        assert not np.any(np.isnan(U))
        assert not np.any(np.isnan(Vt))

    def test_custom_backend(self):
        """A user-supplied callable should work as a backend."""
        rng = np.random.default_rng(0)
        mat = rng.normal(0, 1, (20, 10))

        def dummy_svd(matrix, k):
            U = np.eye(matrix.shape[0], k)
            s = np.ones(k)
            Vt = np.eye(k, matrix.shape[1])
            return U, s, Vt

        U, s, Vt = decompose(mat, k=2, backend=dummy_svd)
        assert U.shape == (20, 2)
        assert s.shape == (2,)

    def test_invalid_k(self, synthetic_lrr):
        with pytest.raises(ValueError, match="k must be >= 1"):
            decompose(synthetic_lrr, k=0)

    def test_k_exceeds_dimensions(self):
        mat = np.ones((3, 5))
        with pytest.raises(ValueError, match="exceeds matrix dimensions"):
            decompose(mat, k=4)

    def test_unknown_backend(self, synthetic_lrr):
        with pytest.raises(ValueError, match="Unknown backend"):
            decompose(synthetic_lrr, k=2, backend="nonexistent")
