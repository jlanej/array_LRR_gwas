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
