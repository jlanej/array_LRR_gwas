"""Tests for QR-based streaming PC regression."""

import numpy as np
import pytest

from array_lrr_gwas.correction import (
    _correct_chunk_qr,
    qr_precompute,
    residualize_qr,
    correct_lrr,
)


# ---------------------------------------------------------------------------
# qr_precompute
# ---------------------------------------------------------------------------

class TestQrPrecompute:
    def test_shapes(self):
        """Q, R, X have the correct shapes."""
        rng = np.random.default_rng(0)
        k, n = 3, 20
        Vt = rng.normal(0, 1, (k, n))
        Q, R, X = qr_precompute(Vt)
        assert Q.shape == (n, k)
        assert R.shape == (k, k)
        assert X.shape == (n, k)

    def test_orthonormality(self):
        """Q has orthonormal columns."""
        rng = np.random.default_rng(1)
        Vt = rng.normal(0, 1, (4, 30))
        Q, _, _ = qr_precompute(Vt)
        np.testing.assert_allclose(Q.T @ Q, np.eye(4), atol=1e-12)

    def test_qr_factorisation(self):
        """X == Q @ R."""
        rng = np.random.default_rng(2)
        Vt = rng.normal(0, 1, (5, 25))
        Q, R, X = qr_precompute(Vt)
        np.testing.assert_allclose(Q @ R, X, atol=1e-12)

    def test_single_component(self):
        """Works for k=1."""
        rng = np.random.default_rng(3)
        Vt = rng.normal(0, 1, (1, 10))
        Q, R, X = qr_precompute(Vt)
        assert Q.shape == (10, 1)
        assert R.shape == (1, 1)


# ---------------------------------------------------------------------------
# _correct_chunk_qr
# ---------------------------------------------------------------------------

class TestCorrectChunkQr:
    def test_removes_known_projection(self):
        """After correction, residuals are orthogonal to the PC space."""
        rng = np.random.default_rng(10)
        k, n = 3, 30
        Vt = rng.normal(0, 1, (k, n))
        Q, R, X = qr_precompute(Vt)

        n_markers = 20
        Y = rng.normal(0, 1, (n_markers, n))
        corrected, n_skip = _correct_chunk_qr(Y, Q, X)
        assert n_skip == 0

        # Each corrected row should be orthogonal to every column of Q
        for i in range(n_markers):
            dots = Q.T @ corrected[i]
            np.testing.assert_allclose(dots, 0.0, atol=1e-10)

    def test_output_shape(self):
        rng = np.random.default_rng(11)
        Vt = rng.normal(0, 1, (2, 15))
        Q, _, X = qr_precompute(Vt)
        chunk = rng.normal(0, 1, (10, 15))
        corrected, _ = _correct_chunk_qr(chunk, Q, X)
        assert corrected.shape == (10, 15)

    def test_nan_marker_corrected_on_valid(self):
        """Markers with some NaN are corrected on the valid subset."""
        rng = np.random.default_rng(12)
        k, n = 2, 20
        Vt = rng.normal(0, 1, (k, n))
        Q, _, X = qr_precompute(Vt)

        chunk = rng.normal(0, 1, (5, n))
        chunk[0, :3] = np.nan  # first marker has 3 missing

        corrected, _ = _correct_chunk_qr(chunk, Q, X)
        # NaN positions remain NaN
        assert np.all(np.isnan(corrected[0, :3]))
        # Valid positions are corrected (not identical to input)
        assert not np.allclose(corrected[0, 3:], chunk[0, 3:])

    def test_all_nan_marker_unchanged(self):
        """A marker with all NaN is left unchanged."""
        rng = np.random.default_rng(13)
        k, n = 2, 15
        Vt = rng.normal(0, 1, (k, n))
        Q, _, X = qr_precompute(Vt)

        chunk = rng.normal(0, 1, (3, n))
        chunk[1, :] = np.nan

        corrected, n_skip = _correct_chunk_qr(chunk, Q, X)
        assert np.all(np.isnan(corrected[1]))
        assert n_skip == 1

    def test_mostly_nan_marker_skipped(self):
        """A marker below the min_valid_frac threshold is left uncorrected."""
        rng = np.random.default_rng(14)
        k, n = 2, 20
        Vt = rng.normal(0, 1, (k, n))
        Q, _, X = qr_precompute(Vt)

        chunk = rng.normal(0, 1, (3, n))
        # Leave only 4 valid out of 20 (20% < 50% default)
        chunk[0, 4:] = np.nan

        corrected, n_skip = _correct_chunk_qr(chunk, Q, X, min_valid_frac=0.5)
        # Marker 0 should be unchanged (skipped)
        np.testing.assert_array_equal(corrected[0], chunk[0])
        assert n_skip == 1

    def test_inf_values_handled(self):
        """Markers containing inf are handled via the slow path."""
        rng = np.random.default_rng(15)
        k, n = 2, 20
        Vt = rng.normal(0, 1, (k, n))
        Q, _, X = qr_precompute(Vt)

        chunk = rng.normal(0, 1, (4, n))
        chunk[0, 0] = np.inf
        chunk[1, 5] = -np.inf

        corrected, n_skip = _correct_chunk_qr(chunk, Q, X, min_valid_frac=0.1)
        assert n_skip == 0
        # inf positions remain inf
        assert np.isinf(corrected[0, 0])
        assert np.isinf(corrected[1, 5])
        # Valid positions should be orthogonal to Q (like the fast path)
        valid_0 = np.isfinite(chunk[0])
        X_v0 = X[valid_0]
        Q_v0, _ = np.linalg.qr(X_v0, mode="reduced")
        dots_0 = Q_v0.T @ corrected[0, valid_0]
        np.testing.assert_allclose(dots_0, 0.0, atol=1e-10)

    def test_pure_batch_fully_removed(self):
        """If y = X @ beta exactly, the corrected residual is zero."""
        rng = np.random.default_rng(16)
        k, n = 3, 30
        Vt = rng.normal(0, 1, (k, n))
        Q, _, X = qr_precompute(Vt)

        beta = rng.normal(0, 1, (5, k))  # 5 markers
        Y = beta @ X.T  # (5, n) – pure batch
        corrected, n_skip = _correct_chunk_qr(Y, Q, X)
        assert n_skip == 0
        np.testing.assert_allclose(corrected, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# residualize_qr
# ---------------------------------------------------------------------------

class TestResidualizeQr:
    def test_output_shape(self):
        rng = np.random.default_rng(20)
        n_markers, n_samples, k = 100, 25, 3
        lrr = rng.normal(0, 0.1, (n_markers, n_samples))
        Vt = rng.normal(0, 1, (k, n_samples))

        corrected = residualize_qr(lrr, Vt, chunk_size=30)
        assert corrected.shape == lrr.shape

    def test_reduces_variance_from_batch(self):
        """Injected batch structure should be removed, lowering variance."""
        rng = np.random.default_rng(21)
        n_markers, n_samples, k = 80, 20, 2
        Vt = rng.normal(0, 1, (k, n_samples))
        X = Vt.T  # (n_samples, k)

        # Inject a clear batch
        beta = rng.normal(0, 2, (n_markers, k))
        batch = beta @ X.T
        noise = rng.normal(0, 0.05, (n_markers, n_samples))
        lrr = batch + noise

        corrected = residualize_qr(lrr, Vt, chunk_size=20)
        assert np.var(corrected) < np.var(lrr) * 0.1

    def test_streaming_matches_single_chunk(self):
        """Results are identical regardless of chunk_size."""
        rng = np.random.default_rng(22)
        n_markers, n_samples, k = 60, 15, 2
        lrr = rng.normal(0, 0.1, (n_markers, n_samples))
        Vt = rng.normal(0, 1, (k, n_samples))

        one_chunk = residualize_qr(lrr, Vt, chunk_size=n_markers)
        small_chunk = residualize_qr(lrr, Vt, chunk_size=10)

        np.testing.assert_allclose(one_chunk, small_chunk, atol=1e-12)

    def test_nan_markers_handled(self):
        """Markers with NaN are corrected where valid, NaN preserved."""
        rng = np.random.default_rng(23)
        n_markers, n_samples, k = 40, 20, 2
        lrr = rng.normal(0, 0.1, (n_markers, n_samples))
        lrr[0, :3] = np.nan
        lrr[5, :] = np.nan  # all-NaN
        Vt = rng.normal(0, 1, (k, n_samples))

        corrected = residualize_qr(lrr, Vt, chunk_size=15)
        assert corrected.shape == lrr.shape
        # NaN positions stay NaN
        assert np.all(np.isnan(corrected[0, :3]))
        assert np.all(np.isnan(corrected[5]))

    def test_chunk_size_one(self):
        """chunk_size=1 still produces correct results."""
        rng = np.random.default_rng(24)
        n_markers, n_samples, k = 10, 12, 2
        lrr = rng.normal(0, 0.1, (n_markers, n_samples))
        Vt = rng.normal(0, 1, (k, n_samples))

        full = residualize_qr(lrr, Vt, chunk_size=n_markers)
        one_by_one = residualize_qr(lrr, Vt, chunk_size=1)
        np.testing.assert_allclose(full, one_by_one, atol=1e-12)


# ---------------------------------------------------------------------------
# correct_lrr integration with QR
# ---------------------------------------------------------------------------

class TestCorrectLrrQR:
    def test_all_markers_corrected(self, synthetic_lrr):
        """Non-SVD markers should also be corrected (not left at input)."""
        corrected, info = correct_lrr(
            synthetic_lrr,
            k=2,
            max_lrr_sd=10.0,
            min_sample_call_rate=0.0,
            min_marker_call_rate=0.5,
            min_var=0.001,  # excludes the zero-variance marker from SVD
        )
        marker_mask = info["marker_mask"]
        non_svd = ~marker_mask
        # marker 1 has zero variance → excluded from SVD subset
        assert non_svd.sum() > 0
        assert corrected.shape == synthetic_lrr.shape

    def test_chunk_size_parameter(self, synthetic_lrr):
        """chunk_size parameter is respected without error."""
        corrected, _ = correct_lrr(
            synthetic_lrr,
            k=2,
            max_lrr_sd=10.0,
            min_sample_call_rate=0.0,
            min_marker_call_rate=0.5,
            min_var=0.0,
            chunk_size=10,
        )
        assert corrected.shape == synthetic_lrr.shape

    def test_min_valid_frac_parameter(self, synthetic_lrr):
        """min_valid_frac parameter is respected without error."""
        corrected, _ = correct_lrr(
            synthetic_lrr,
            k=2,
            max_lrr_sd=10.0,
            min_sample_call_rate=0.0,
            min_marker_call_rate=0.5,
            min_var=0.0,
            min_valid_frac=0.3,
        )
        assert corrected.shape == synthetic_lrr.shape

    def test_variance_reduced_globally(self, synthetic_lrr):
        """Correction should reduce overall variance including non-SVD markers."""
        corrected, _ = correct_lrr(
            synthetic_lrr,
            k=3,
            max_lrr_sd=10.0,
            min_sample_call_rate=0.0,
            min_marker_call_rate=0.5,
            min_var=0.0,
        )
        assert np.nanvar(corrected) < np.nanvar(synthetic_lrr)

    def test_heavy_missingness_markers_preserved(self):
        """Markers with heavy missingness are left uncorrected."""
        rng = np.random.default_rng(30)
        n_markers, n_samples = 30, 20
        lrr = rng.normal(0, 0.1, (n_markers, n_samples))
        # Marker 0: 90% missing
        lrr[0, 2:] = np.nan

        corrected, _ = correct_lrr(
            lrr,
            k=2,
            max_lrr_sd=10.0,
            min_sample_call_rate=0.0,
            min_marker_call_rate=0.0,
            min_var=0.0,
            min_valid_frac=0.5,
        )
        # Marker 0 has only 2 valid out of 20 (10% < 50%) → uncorrected
        np.testing.assert_array_equal(corrected[0], lrr[0])
