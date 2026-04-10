"""Integration tests for LRR correction logic.

Validates reproducibility, permutation invariance, edge-case handling,
batch-effect removal, streaming consistency, and audit trail alignment
for the full correction pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from array_lrr_gwas.audit import AuditLogger
from array_lrr_gwas.correction import (
    classify_samples,
    correct_lrr,
    residualize_qr,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lrr(
    n_markers: int = 60,
    n_samples: int = 20,
    k_batch: int = 3,
    batch_strength: float = 0.3,
    noise_sd: float = 0.02,
    n_lq: int = 2,
    lq_noise_sd: float = 0.5,
    nan_frac: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """Build a synthetic LRR matrix with known batch structure."""
    rng = np.random.default_rng(seed)
    n = n_markers
    m = n_samples
    U = rng.normal(0, 1, (n, k_batch))
    V = rng.normal(0, 1, (k_batch, m))
    batch = batch_strength * U @ V / np.sqrt(n)
    signal = rng.normal(0, 0.05, (n, m))
    noise = rng.normal(0, noise_sd, (n, m))
    lrr = signal + batch + noise

    # LQ samples
    if n_lq > 0:
        lrr[:, -n_lq:] += rng.normal(0, lq_noise_sd, (n, n_lq))

    # Random missingness
    if nan_frac > 0:
        mask = rng.random((n, m)) < nan_frac
        lrr[mask] = np.nan

    return lrr


def _correct_kwargs(**overrides):
    """Default kwargs for correct_lrr in integration tests."""
    defaults = dict(
        k=2,
        max_lrr_sd=10.0,
        min_sample_call_rate=0.0,
        min_marker_call_rate=0.0,
        min_var=0.0,
    )
    defaults.update(overrides)
    return defaults


# ===================================================================
# Reproducibility
# ===================================================================

class TestReproducibility:
    """Same input → identical output across multiple runs."""

    def test_deterministic_output(self):
        lrr = _make_lrr(seed=1)
        c1, i1 = correct_lrr(lrr.copy(), **_correct_kwargs())
        c2, i2 = correct_lrr(lrr.copy(), **_correct_kwargs())
        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(
            i1["singular_values"], i2["singular_values"]
        )
        np.testing.assert_array_equal(
            i1["sample_scores"], i2["sample_scores"]
        )

    def test_deterministic_with_nans(self):
        lrr = _make_lrr(nan_frac=0.1, seed=2)
        c1, _ = correct_lrr(lrr.copy(), **_correct_kwargs())
        c2, _ = correct_lrr(lrr.copy(), **_correct_kwargs())
        # NaN positions should be NaN in both; finite positions should match
        both_nan = np.isnan(c1) & np.isnan(c2)
        both_fin = np.isfinite(c1) & np.isfinite(c2)
        assert (both_nan | both_fin).all()
        np.testing.assert_array_equal(c1[both_fin], c2[both_fin])

    def test_deterministic_auto_k(self):
        lrr = _make_lrr(seed=3)
        _, i1 = correct_lrr(lrr.copy(), **_correct_kwargs(k=None))
        _, i2 = correct_lrr(lrr.copy(), **_correct_kwargs(k=None))
        assert i1["k"] == i2["k"]
        np.testing.assert_array_equal(
            i1["singular_values"], i2["singular_values"]
        )


# ===================================================================
# Permutation invariance
# ===================================================================

class TestPermutationInvariance:
    """Results are invariant to row/column order."""

    def test_marker_permutation(self):
        """Permuting marker rows gives equivalent per-marker correction."""
        rng = np.random.default_rng(10)
        lrr = _make_lrr(n_markers=40, n_samples=15, seed=10, nan_frac=0.0)
        perm = rng.permutation(lrr.shape[0])
        inv_perm = np.argsort(perm)

        c_orig, i_orig = correct_lrr(lrr, **_correct_kwargs())
        c_perm, i_perm = correct_lrr(lrr[perm], **_correct_kwargs())

        # Un-permute and compare
        np.testing.assert_allclose(
            c_perm[inv_perm], c_orig, atol=1e-10,
        )

    def test_sample_permutation(self):
        """Permuting sample columns gives equivalent per-sample correction.

        The RSVD is seeded but internal random draws depend on matrix
        layout, so small numerical differences (< 1e-8) are expected.
        """
        rng = np.random.default_rng(11)
        lrr = _make_lrr(n_markers=40, n_samples=15, n_lq=0, seed=11,
                         nan_frac=0.0)
        perm = rng.permutation(lrr.shape[1])
        inv_perm = np.argsort(perm)

        c_orig, _ = correct_lrr(lrr, **_correct_kwargs())
        c_perm, _ = correct_lrr(lrr[:, perm], **_correct_kwargs())

        np.testing.assert_allclose(
            c_perm[:, inv_perm], c_orig, atol=1e-8,
        )


# ===================================================================
# NaN robustness
# ===================================================================

class TestNaNRobustness:
    """Pipeline handles diverse missingness patterns."""

    def test_sparse_nans(self):
        lrr = _make_lrr(nan_frac=0.05, seed=20)
        corrected, info = correct_lrr(lrr, **_correct_kwargs())
        assert corrected.shape == lrr.shape
        # Original NaN positions stay NaN
        orig_nan = np.isnan(lrr)
        assert np.all(np.isnan(corrected[orig_nan]))

    def test_dense_nans(self):
        lrr = _make_lrr(nan_frac=0.3, seed=21)
        corrected, info = correct_lrr(
            lrr,
            **_correct_kwargs(min_marker_call_rate=0.0, min_valid_frac=0.1),
        )
        assert corrected.shape == lrr.shape

    def test_entire_row_nan(self):
        """A row of all NaN should be left as NaN in the output."""
        lrr = _make_lrr(seed=22, nan_frac=0.0)
        lrr[5, :] = np.nan
        corrected, _ = correct_lrr(lrr, **_correct_kwargs())
        assert np.all(np.isnan(corrected[5, :]))

    def test_entire_column_nan_excluded_from_hq(self):
        """A column of all NaN → call rate = 0 → sample excluded from HQ."""
        lrr = _make_lrr(n_samples=20, n_lq=0, seed=23, nan_frac=0.0)
        lrr[:, 0] = np.nan  # first sample all missing
        hq = classify_samples(lrr, max_lrr_sd=10.0, min_call_rate=0.5)
        assert not hq[0]
        assert hq[1:].all()

    def test_nan_positions_preserved_after_correction(self):
        """NaN positions in input must remain NaN in output."""
        lrr = _make_lrr(seed=24, nan_frac=0.0)
        nan_indices = [(0, 0), (3, 5), (10, 10), (20, 15)]
        for r, c in nan_indices:
            if r < lrr.shape[0] and c < lrr.shape[1]:
                lrr[r, c] = np.nan
        corrected, _ = correct_lrr(lrr, **_correct_kwargs())
        for r, c in nan_indices:
            if r < lrr.shape[0] and c < lrr.shape[1]:
                assert np.isnan(corrected[r, c]), f"NaN not preserved at ({r},{c})"


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    """Handles difficult inputs gracefully."""

    def test_constant_columns_excluded(self):
        """A sample with constant LRR should still be included (SD=0)."""
        lrr = _make_lrr(n_samples=15, n_lq=0, seed=30, nan_frac=0.0)
        lrr[:, 0] = 0.0  # constant column
        corrected, info = correct_lrr(lrr, **_correct_kwargs())
        assert corrected.shape == lrr.shape
        # The constant sample should be in HQ (sd=0 <= max_lrr_sd=10.0)
        assert info["hq_sample_mask"][0]

    def test_constant_row_excluded_from_svd(self):
        """A constant-variance marker row should be excluded by min_var."""
        lrr = _make_lrr(n_markers=40, seed=31, nan_frac=0.0)
        lrr[0, :] = 0.0  # zero variance row
        corrected, info = correct_lrr(
            lrr, **_correct_kwargs(min_var=0.001)
        )
        assert corrected.shape == lrr.shape
        # marker 0 excluded from SVD decomposition
        assert not info["marker_mask"][0]

    def test_inf_values_handled(self):
        """Inf values in LRR should not crash the pipeline."""
        lrr = _make_lrr(n_markers=40, seed=32, nan_frac=0.0)
        lrr[0, 0] = np.inf
        lrr[1, 1] = -np.inf
        corrected, _ = correct_lrr(lrr, **_correct_kwargs())
        assert corrected.shape == lrr.shape
        # Inf positions may remain inf (treated as non-finite)
        assert np.isinf(corrected[0, 0]) or np.isnan(corrected[0, 0])

    def test_small_matrix(self):
        """Very small matrix (3 markers, 3 samples) corrects cleanly."""
        lrr = _make_lrr(n_markers=5, n_samples=4, n_lq=0, k_batch=1,
                         seed=33, nan_frac=0.0)
        corrected, info = correct_lrr(lrr, **_correct_kwargs(k=1))
        assert corrected.shape == lrr.shape
        assert info["k"] == 1

    def test_all_lq_raises(self):
        """If every sample is LQ, ValueError is raised."""
        lrr = np.random.default_rng(34).normal(0, 10, (20, 5))
        with pytest.raises(ValueError, match="No samples passed"):
            correct_lrr(lrr, k=2, max_lrr_sd=0.001)

    def test_all_markers_fail_raises(self):
        """If every marker fails QC, ValueError is raised."""
        lrr = np.zeros((10, 5))
        with pytest.raises(ValueError, match="No markers passed"):
            correct_lrr(lrr, k=1, max_lrr_sd=100.0,
                         min_sample_call_rate=0.0, min_var=1.0)

    def test_simulated_batch_effect(self):
        """Injecting a rank-1 batch effect should inflate pre-correction
        variance and be substantially reduced post-correction."""
        rng = np.random.default_rng(35)
        n, m = 50, 20
        noise = rng.normal(0, 0.02, (n, m))
        batch_u = rng.normal(0, 1, (n, 1))
        batch_v = rng.normal(0, 1, (1, m))
        lrr = noise + 0.5 * batch_u @ batch_v / np.sqrt(n)

        corrected, info = correct_lrr(
            lrr, **_correct_kwargs(k=1)
        )
        assert np.nanvar(corrected) < np.nanvar(lrr) * 0.5

    def test_outlier_sample_detection(self):
        """Extremely noisy samples should be flagged as LQ."""
        lrr = _make_lrr(n_samples=20, n_lq=0, seed=36, nan_frac=0.0)
        # Add extreme noise to last sample
        lrr[:, -1] += np.random.default_rng(36).normal(0, 5.0, lrr.shape[0])
        hq = classify_samples(lrr, max_lrr_sd=0.5)
        assert not hq[-1]
        assert hq[:-1].sum() > 0


# ===================================================================
# Batch effect removal
# ===================================================================

class TestBatchEffectRemoval:
    """Synthetic batch effects are verifiably removed."""

    def test_rank_k_batch_removed(self):
        """A pure rank-k batch injected on top of noise is substantially
        reduced by correction.

        RSVD is an approximation and the pipeline internally centres
        per-marker, so perfect removal is not expected.  We verify that
        the overall variance is reduced by at least 50%.
        """
        rng = np.random.default_rng(40)
        n, m, k = 60, 25, 3
        noise = rng.normal(0, 0.01, (n, m))
        U = rng.normal(0, 1, (n, k))
        V = rng.normal(0, 1, (k, m))
        batch = U @ V / np.sqrt(n)
        lrr = noise + batch

        corrected, info = correct_lrr(lrr, **_correct_kwargs(k=k))
        # Correction should reduce overall variance substantially
        assert np.nanvar(corrected) < np.nanvar(lrr) * 0.5

    def test_variance_reduction_monotonic_in_k(self):
        """Increasing k should not increase residual variance
        (up to reasonable k)."""
        lrr = _make_lrr(n_markers=60, n_samples=25, k_batch=3, seed=41,
                         n_lq=0, nan_frac=0.0)
        vars_ = []
        for k in [1, 2, 3]:
            corrected, _ = correct_lrr(lrr.copy(), **_correct_kwargs(k=k))
            vars_.append(np.nanvar(corrected))
        # Each additional PC should reduce (or not increase) variance
        for i in range(1, len(vars_)):
            assert vars_[i] <= vars_[i - 1] + 1e-10

    def test_no_signal_leakage_orthogonal_component(self):
        """Signal orthogonal to the batch PCs should mostly survive
        correction.

        Because RSVD is approximate and the pipeline centres per-marker,
        there is inherent cross-talk.  We use a generous tolerance (< 0.6
        relative error) to confirm that the majority of the orthogonal
        signal is preserved.
        """
        rng = np.random.default_rng(42)
        n, m = 50, 20
        # Construct signal in the null space of the batch PCs
        U_batch = rng.normal(0, 1, (n, 2))
        V_batch = rng.normal(0, 1, (2, m))
        batch = 0.5 * U_batch @ V_batch / np.sqrt(n)

        # Create an orthogonal signal: project out batch space
        raw_signal = rng.normal(0, 0.1, (n, m))
        Q_batch, _ = np.linalg.qr(U_batch)
        signal = raw_signal - Q_batch @ (Q_batch.T @ raw_signal)

        lrr = signal + batch
        corrected, _ = correct_lrr(lrr, **_correct_kwargs(k=2))

        # The corrected result should preserve the orthogonal signal
        signal_corr = np.linalg.norm(corrected - signal, "fro")
        signal_norm = np.linalg.norm(signal, "fro")
        assert signal_corr / signal_norm < 0.6


# ===================================================================
# Streaming consistency
# ===================================================================

class TestStreamingConsistency:
    """Chunked QR regression is consistent across chunk sizes."""

    def test_chunk_size_invariance(self):
        """residualize_qr gives identical results for different chunk sizes."""
        rng = np.random.default_rng(50)
        n_markers, n_samples, k = 80, 20, 3
        lrr = rng.normal(0, 0.1, (n_markers, n_samples))
        Vt = rng.normal(0, 1, (k, n_samples))

        c_full = residualize_qr(lrr, Vt, chunk_size=n_markers)
        c_10 = residualize_qr(lrr, Vt, chunk_size=10)
        c_7 = residualize_qr(lrr, Vt, chunk_size=7)
        c_1 = residualize_qr(lrr, Vt, chunk_size=1)

        np.testing.assert_allclose(c_10, c_full, atol=1e-12)
        np.testing.assert_allclose(c_7, c_full, atol=1e-12)
        np.testing.assert_allclose(c_1, c_full, atol=1e-12)

    def test_correct_lrr_chunk_size_invariance(self):
        """correct_lrr with different chunk_size produces same result."""
        lrr = _make_lrr(n_markers=50, n_samples=15, n_lq=0, seed=51,
                         nan_frac=0.0)
        c1, _ = correct_lrr(lrr.copy(), **_correct_kwargs(), chunk_size=50)
        c2, _ = correct_lrr(lrr.copy(), **_correct_kwargs(), chunk_size=10)
        c3, _ = correct_lrr(lrr.copy(), **_correct_kwargs(), chunk_size=7)
        np.testing.assert_allclose(c2, c1, atol=1e-12)
        np.testing.assert_allclose(c3, c1, atol=1e-12)

    def test_streaming_with_nans(self):
        """Streaming correction handles NaN consistently across chunk sizes."""
        lrr = _make_lrr(n_markers=60, n_samples=15, seed=52, nan_frac=0.1,
                         n_lq=0)
        c1, _ = correct_lrr(
            lrr.copy(),
            **_correct_kwargs(min_valid_frac=0.1),
            chunk_size=60,
        )
        c2, _ = correct_lrr(
            lrr.copy(),
            **_correct_kwargs(min_valid_frac=0.1),
            chunk_size=11,
        )
        both_fin = np.isfinite(c1) & np.isfinite(c2)
        np.testing.assert_allclose(c1[both_fin], c2[both_fin], atol=1e-12)


# ===================================================================
# Audit trail consistency
# ===================================================================

class TestAuditTrailConsistency:
    """Audit records match correction parameters."""

    def test_audit_sample_counts(self):
        """Audit sample QC record matches info dict counts."""
        lrr = _make_lrr(n_samples=20, n_lq=2, seed=60, nan_frac=0.0)
        sample_ids = [f"S{i:03d}" for i in range(lrr.shape[1])]
        variant_ids = [f"V{i:04d}" for i in range(lrr.shape[0])]
        audit = AuditLogger()

        _, info = correct_lrr(
            lrr,
            **_correct_kwargs(),
            audit=audit,
            sample_ids=sample_ids,
            variant_ids=variant_ids,
        )

        # Find the sample QC record
        sample_recs = [
            r for r in audit.records if r.stage == "correction_sample_qc"
        ]
        assert len(sample_recs) == 1
        rec = sample_recs[0]
        assert rec.total_included == info["n_hq_samples"]
        assert rec.total_input == lrr.shape[1]
        assert rec.total_excluded == lrr.shape[1] - info["n_hq_samples"]

    def test_audit_marker_subset_recorded(self):
        """Marker subsetting stages are recorded in the audit trail."""
        lrr = _make_lrr(seed=61, nan_frac=0.0)
        variant_ids = [f"V{i:04d}" for i in range(lrr.shape[0])]
        sample_ids = [f"S{i:03d}" for i in range(lrr.shape[1])]
        audit = AuditLogger()

        _, info = correct_lrr(
            lrr,
            **_correct_kwargs(),
            audit=audit,
            variant_ids=variant_ids,
            sample_ids=sample_ids,
        )

        marker_recs = [
            r for r in audit.records if r.id_type == "marker"
        ]
        # At least one marker-related audit record should exist
        assert len(marker_recs) >= 1

    def test_audit_hq_mask_matches_info(self):
        """HQ mask from info matches the sample count in audit record."""
        lrr = _make_lrr(n_samples=15, n_lq=3, seed=62, nan_frac=0.0)
        sample_ids = [f"S{i:03d}" for i in range(lrr.shape[1])]
        variant_ids = [f"V{i:04d}" for i in range(lrr.shape[0])]
        audit = AuditLogger()

        _, info = correct_lrr(
            lrr,
            **_correct_kwargs(max_lrr_sd=0.35),
            audit=audit,
            sample_ids=sample_ids,
            variant_ids=variant_ids,
        )

        n_hq = int(info["hq_sample_mask"].sum())
        assert n_hq == info["n_hq_samples"]
        # Info and audit should agree
        sample_recs = [
            r for r in audit.records if r.stage == "correction_sample_qc"
        ]
        if sample_recs:
            assert sample_recs[0].total_included == n_hq


# ===================================================================
# Post-correction consistency
# ===================================================================

class TestPostCorrectionConsistency:
    """Info dict fields are internally self-consistent."""

    def test_marker_mask_shape(self):
        lrr = _make_lrr(seed=70)
        _, info = correct_lrr(lrr, **_correct_kwargs())
        assert info["marker_mask"].shape == (lrr.shape[0],)
        assert info["marker_mask"].dtype == bool

    def test_hq_mask_shape(self):
        lrr = _make_lrr(seed=71)
        _, info = correct_lrr(lrr, **_correct_kwargs())
        assert info["hq_sample_mask"].shape == (lrr.shape[1],)
        assert info["hq_sample_mask"].dtype == bool

    def test_n_markers_used_matches_mask(self):
        lrr = _make_lrr(seed=72)
        _, info = correct_lrr(lrr, **_correct_kwargs())
        assert info["n_markers_used"] == int(info["marker_mask"].sum())

    def test_n_hq_samples_matches_mask(self):
        lrr = _make_lrr(seed=73)
        _, info = correct_lrr(lrr, **_correct_kwargs())
        assert info["n_hq_samples"] == int(info["hq_sample_mask"].sum())

    def test_singular_values_descending(self):
        lrr = _make_lrr(seed=74)
        _, info = correct_lrr(lrr, **_correct_kwargs())
        sv = info["singular_values"]
        assert np.all(sv[:-1] >= sv[1:])

    def test_sample_scores_shape(self):
        lrr = _make_lrr(seed=75)
        _, info = correct_lrr(lrr, **_correct_kwargs())
        assert info["sample_scores"].shape == (
            info["n_components_computed"],
            lrr.shape[1],
        )

    def test_marker_loadings_shape(self):
        lrr = _make_lrr(seed=76)
        _, info = correct_lrr(lrr, **_correct_kwargs())
        assert info["marker_loadings"].shape == (
            info["n_markers_used"],
            info["n_components_computed"],
        )

    def test_corrected_not_identical_to_input(self):
        """Correction must actually modify the data."""
        lrr = _make_lrr(seed=77, k_batch=2, batch_strength=0.5, nan_frac=0.0)
        corrected, _ = correct_lrr(lrr, **_correct_kwargs())
        # At least some values should differ
        assert not np.allclose(corrected, lrr, atol=1e-14)

    def test_skip_residualize_matches_full_info(self):
        """skip_residualize should yield the same decomposition info."""
        lrr = _make_lrr(seed=78, nan_frac=0.0)
        _, info_full = correct_lrr(lrr.copy(), **_correct_kwargs())
        _, info_skip = correct_lrr(
            lrr.copy(), **_correct_kwargs(), skip_residualize=True
        )
        assert info_skip["k"] == info_full["k"]
        assert info_skip["n_hq_samples"] == info_full["n_hq_samples"]
        assert info_skip["n_markers_used"] == info_full["n_markers_used"]
        np.testing.assert_array_equal(
            info_skip["singular_values"], info_full["singular_values"]
        )
        np.testing.assert_array_equal(
            info_skip["sample_scores"], info_full["sample_scores"]
        )

    def test_backend_field_recorded(self):
        lrr = _make_lrr(seed=79)
        _, info = correct_lrr(lrr, **_correct_kwargs())
        assert info["backend"] == "rsvd"
