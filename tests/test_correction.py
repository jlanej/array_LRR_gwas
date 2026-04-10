"""Tests for the end-to-end correction pipeline."""

import numpy as np
import pytest

from array_lrr_gwas.correction import (
    classify_samples,
    correct_lrr,
    extrapolate_pcs,
    residualize,
)


class TestClassifySamples:
    def test_hq_lq_split(self, synthetic_lrr):
        hq = classify_samples(synthetic_lrr, max_lrr_sd=0.35)
        # Last 2 samples have SD >> 0.35
        assert not hq[-1]
        assert not hq[-2]
        # Earlier samples should mostly be HQ
        assert hq[:10].sum() > 0

    def test_all_hq_with_loose_threshold(self, synthetic_lrr):
        hq = classify_samples(synthetic_lrr, max_lrr_sd=100.0, min_call_rate=0.0)
        assert hq.all()


class TestExtrapolatePcs:
    def test_output_shape(self):
        rng = np.random.default_rng(0)
        lrr_lq = rng.normal(0, 0.1, (30, 5))
        row_means = np.zeros((30, 1))
        U = rng.normal(0, 1, (30, 3))
        s = np.array([10.0, 5.0, 2.0])
        Vt_lq = extrapolate_pcs(lrr_lq, row_means, U, s)
        assert Vt_lq.shape == (3, 5)


class TestResidualize:
    def test_removes_component(self):
        rng = np.random.default_rng(0)
        n, m = 40, 15
        U = rng.normal(0, 1, (n, 2))
        s = np.array([10.0, 5.0])
        Vt = rng.normal(0, 1, (2, m))
        batch = U @ np.diag(s) @ Vt
        noise = rng.normal(0, 0.01, (n, m))
        lrr = batch + noise

        corrected = residualize(lrr, U, s, Vt)
        # After removing the exact batch, residual should be close to noise
        np.testing.assert_allclose(corrected, noise, atol=1e-10)


class TestCorrectLrr:
    def test_basic_pipeline(self, synthetic_lrr):
        corrected, info = correct_lrr(
            synthetic_lrr,
            k=2,
            max_lrr_sd=10.0,  # let all samples be HQ for simplicity
            min_sample_call_rate=0.0,
            min_marker_call_rate=0.5,
            min_var=0.0,
        )
        assert corrected.shape == synthetic_lrr.shape
        assert info["k"] == 2
        assert "singular_values" in info

    def test_auto_k(self, synthetic_lrr):
        corrected, info = correct_lrr(
            synthetic_lrr,
            k=None,
            max_lrr_sd=10.0,
            min_sample_call_rate=0.0,
            min_marker_call_rate=0.5,
            min_var=0.0,
        )
        assert info["k"] >= 1
        assert corrected.shape == synthetic_lrr.shape
        # n_components_computed must be >= k (pilot covers at least k)
        assert info["n_components_computed"] >= info["k"]
        # singular_values and sample_scores reflect all computed components
        assert len(info["singular_values"]) == info["n_components_computed"]
        assert info["sample_scores"].shape[0] == info["n_components_computed"]
        assert info["marker_loadings"].shape[1] == info["n_components_computed"]

    def test_auto_k_all_components_in_output(self, synthetic_lrr):
        """When n_components > k, all computed PCs are stored in info."""
        corrected, info = correct_lrr(
            synthetic_lrr,
            k=None,
            n_components=3,
            max_lrr_sd=10.0,
            min_sample_call_rate=0.0,
            min_marker_call_rate=0.5,
            min_var=0.0,
        )
        assert info["n_components_computed"] == 3
        assert len(info["singular_values"]) == 3
        assert info["sample_scores"].shape[0] == 3
        assert info["marker_loadings"].shape[1] == 3
        # But only k were used for correction
        assert info["k"] <= 3

    def test_explicit_k_components_computed_equals_k(self, synthetic_lrr):
        """When k is explicit, n_components_computed == k."""
        _, info = correct_lrr(
            synthetic_lrr,
            k=2,
            max_lrr_sd=10.0,
            min_sample_call_rate=0.0,
            min_marker_call_rate=0.5,
            min_var=0.0,
        )
        assert info["n_components_computed"] == 2
        assert len(info["singular_values"]) == 2
        assert info["sample_scores"].shape[0] == 2
        assert info["marker_loadings"].shape[1] == 2

    def test_auto_k_accepts_n_components(self, synthetic_lrr):
        corrected, info = correct_lrr(
            synthetic_lrr,
            k=None,
            n_components=3,
            max_lrr_sd=10.0,
            min_sample_call_rate=0.0,
            min_marker_call_rate=0.5,
            min_var=0.0,
        )
        assert info["k"] >= 1
        assert corrected.shape == synthetic_lrr.shape

    def test_auto_k_invalid_n_components_raises(self, synthetic_lrr):
        with pytest.raises(ValueError, match="n_components must be >= 1"):
            correct_lrr(
                synthetic_lrr,
                k=None,
                n_components=0,
                max_lrr_sd=10.0,
                min_sample_call_rate=0.0,
                min_marker_call_rate=0.5,
                min_var=0.0,
            )

    def test_variance_reduced(self, synthetic_lrr):
        corrected, _ = correct_lrr(
            synthetic_lrr,
            k=3,
            max_lrr_sd=10.0,
            min_sample_call_rate=0.0,
            min_marker_call_rate=0.5,
            min_var=0.0,
        )
        # Correction should reduce overall variance
        orig_var = np.nanvar(synthetic_lrr)
        corr_var = np.nanvar(corrected)
        assert corr_var < orig_var

    def test_no_hq_samples_raises(self):
        lrr = np.random.default_rng(0).normal(0, 10, (10, 5))
        with pytest.raises(ValueError, match="No samples passed"):
            correct_lrr(lrr, k=2, max_lrr_sd=0.001)

    def test_no_markers_raises(self):
        lrr = np.zeros((10, 5))  # all zero variance
        with pytest.raises(ValueError, match="No markers passed"):
            correct_lrr(
                lrr,
                k=2,
                max_lrr_sd=100.0,
                min_sample_call_rate=0.0,
                min_var=0.1,
            )

    def test_with_hq_lq_split(self, synthetic_lrr):
        """When there are both HQ and LQ samples, PCs are extrapolated."""
        corrected, info = correct_lrr(
            synthetic_lrr,
            k=2,
            max_lrr_sd=0.35,
            min_sample_call_rate=0.0,
            min_marker_call_rate=0.5,
            min_var=0.0,
        )
        assert info["n_hq_samples"] < synthetic_lrr.shape[1]
        assert corrected.shape == synthetic_lrr.shape

    def test_upstream_qc_mask_reduces_markers(self, synthetic_lrr):
        """upstream_qc_mask excludes additional markers from the decomposition."""
        # Baseline without QC mask
        _, info_base = correct_lrr(
            synthetic_lrr,
            k=2,
            max_lrr_sd=10.0,
            min_sample_call_rate=0.0,
            min_marker_call_rate=0.5,
            min_var=0.0,
        )
        # QC mask that excludes half the markers
        n_markers = synthetic_lrr.shape[0]
        qc_mask = np.array([i % 2 == 0 for i in range(n_markers)], dtype=bool)
        corrected, info_qc = correct_lrr(
            synthetic_lrr,
            k=2,
            max_lrr_sd=10.0,
            min_sample_call_rate=0.0,
            min_marker_call_rate=0.5,
            min_var=0.0,
            upstream_qc_mask=qc_mask,
        )
        assert corrected.shape == synthetic_lrr.shape
        assert info_qc["n_markers_used"] < info_base["n_markers_used"]

    def test_upstream_qc_mask_none_unchanged(self, synthetic_lrr):
        """upstream_qc_mask=None does not change the marker count."""
        _, info_none = correct_lrr(
            synthetic_lrr,
            k=2,
            max_lrr_sd=10.0,
            min_sample_call_rate=0.0,
            min_marker_call_rate=0.5,
            min_var=0.0,
            upstream_qc_mask=None,
        )
        _, info_no_arg = correct_lrr(
            synthetic_lrr,
            k=2,
            max_lrr_sd=10.0,
            min_sample_call_rate=0.0,
            min_marker_call_rate=0.5,
            min_var=0.0,
        )
        assert info_none["n_markers_used"] == info_no_arg["n_markers_used"]

    def test_sample_classification_uses_autosomal_only(self, synthetic_lrr):
        """Sample HQ/LQ classification must ignore non-autosomal markers.

        If a sex-chromosome marker has inf/-inf LRR, classify_samples must
        not see it — only the autosomal subset is passed for QC.
        """
        n_markers, n_samples = synthetic_lrr.shape
        # Build chromosome labels: all autosomes except last 5 rows (chrY)
        chroms = np.array(["chr1"] * (n_markers - 5) + ["chrY"] * 5)

        # Inject -inf into every chrY marker for every sample.
        # Without autosomal filtering classify_samples would see inf and
        # np.nanstd would return nan → no HQ samples → ValueError.
        lrr_with_inf = synthetic_lrr.copy()
        lrr_with_inf[-5:, :] = np.inf

        corrected, info = correct_lrr(
            lrr_with_inf,
            chromosomes=chroms,
            k=2,
            max_lrr_sd=10.0,
            min_sample_call_rate=0.0,
            min_marker_call_rate=0.5,
            min_var=0.0,
        )
        # Correction must succeed and output shape must match input
        assert corrected.shape == lrr_with_inf.shape
        assert info["n_hq_samples"] > 0

    def test_ram_budget_triggers_subsample(self, synthetic_lrr):
        """With a very small max_ram_gb, marker count should be reduced."""
        # Run without budget
        _, info_no_budget = correct_lrr(
            synthetic_lrr,
            k=2,
            max_lrr_sd=10.0,
            min_sample_call_rate=0.0,
            min_marker_call_rate=0.5,
            min_var=0.0,
        )

        # Run with an extremely small RAM budget to force subsampling
        # synthetic_lrr is 50×15; need a budget that forces < 48 markers
        # Budget formula: n_markers = max_ram_bytes / (2.5 * n_samples * 8)
        # For 15 samples: need max_ram_bytes < 48 * 2.5 * 15 * 8 = 14400
        # So use ~5000 bytes (about 5e-6 GB) → budget ≈ 16 markers
        tiny_gb = 5000 / 1024**3
        n_markers, n_samples = synthetic_lrr.shape
        chroms = np.array(["chr1"] * n_markers)
        pos = np.arange(n_markers, dtype=np.intp) * 1000

        corrected, info_budgeted = correct_lrr(
            synthetic_lrr,
            chromosomes=chroms,
            positions=pos,
            k=2,
            max_lrr_sd=10.0,
            min_sample_call_rate=0.0,
            min_marker_call_rate=0.5,
            min_var=0.0,
            max_ram_gb=tiny_gb,
        )
        assert corrected.shape == synthetic_lrr.shape
        assert info_budgeted["n_markers_used"] < info_no_budget["n_markers_used"]
        assert info_budgeted["rsvd_subsampled"] is True
        assert info_budgeted["rsvd_marker_budget"] is not None

    def test_ram_budget_no_subsample_needed(self, synthetic_lrr):
        """With a large max_ram_gb, marker count should be unchanged."""
        _, info_no_budget = correct_lrr(
            synthetic_lrr,
            k=2,
            max_lrr_sd=10.0,
            min_sample_call_rate=0.0,
            min_marker_call_rate=0.5,
            min_var=0.0,
        )

        _, info_large = correct_lrr(
            synthetic_lrr,
            k=2,
            max_lrr_sd=10.0,
            min_sample_call_rate=0.0,
            min_marker_call_rate=0.5,
            min_var=0.0,
            max_ram_gb=100.0,
        )
        assert info_large["n_markers_used"] == info_no_budget["n_markers_used"]
        assert info_large["rsvd_subsampled"] is False

    def test_ram_budget_none_no_subsample(self, synthetic_lrr):
        """max_ram_gb=None produces no subsampling metadata."""
        _, info = correct_lrr(
            synthetic_lrr,
            k=2,
            max_lrr_sd=10.0,
            min_sample_call_rate=0.0,
            min_marker_call_rate=0.5,
            min_var=0.0,
            max_ram_gb=None,
        )
        assert info["rsvd_subsampled"] is False
        assert info["rsvd_marker_budget"] is None

    def test_skip_residualize(self, synthetic_lrr):
        """skip_residualize=True returns None for corrected and valid info."""
        corrected, info = correct_lrr(
            synthetic_lrr,
            k=2,
            max_lrr_sd=10.0,
            min_sample_call_rate=0.0,
            min_marker_call_rate=0.5,
            min_var=0.0,
            skip_residualize=True,
        )
        assert corrected is None
        assert info["k"] == 2
        assert "singular_values" in info
        assert "sample_scores" in info
        assert info["sample_scores"].shape[0] >= info["k"]
        assert info["sample_scores"].shape[1] == synthetic_lrr.shape[1]

    def test_skip_residualize_info_matches(self, synthetic_lrr):
        """skip_residualize produces the same info as a full correction."""
        _, info_full = correct_lrr(
            synthetic_lrr,
            k=2,
            max_lrr_sd=10.0,
            min_sample_call_rate=0.0,
            min_marker_call_rate=0.5,
            min_var=0.0,
        )
        _, info_skip = correct_lrr(
            synthetic_lrr,
            k=2,
            max_lrr_sd=10.0,
            min_sample_call_rate=0.0,
            min_marker_call_rate=0.5,
            min_var=0.0,
            skip_residualize=True,
        )
        assert info_skip["k"] == info_full["k"]
        assert info_skip["n_hq_samples"] == info_full["n_hq_samples"]
        assert info_skip["n_markers_used"] == info_full["n_markers_used"]
        np.testing.assert_array_equal(
            info_skip["singular_values"],
            info_full["singular_values"],
        )
        np.testing.assert_array_equal(
            info_skip["sample_scores"],
            info_full["sample_scores"],
        )
