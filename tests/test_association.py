"""Tests for the GWAS association engine (array_lrr_gwas.association)."""

from __future__ import annotations

import numpy as np
import pytest

from array_lrr_gwas.association import (
    AssociationResult,
    _ols_scan,
    _logistic_scan,
    run_association,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_variants(n: int, chrom: str = "chr1") -> list[dict]:
    """Create minimal variant metadata for *n* markers."""
    return [{"chrom": chrom, "pos": i + 1, "id": f"snp{i}"} for i in range(n)]


# ---------------------------------------------------------------------------
# OLS scan
# ---------------------------------------------------------------------------

class TestOlsScan:
    """Tests for the vectorised OLS association scan."""

    def test_output_shapes(self, synthetic_lrr: np.ndarray) -> None:
        n_markers, n_samples = synthetic_lrr.shape
        phenotype = np.random.default_rng(42).normal(size=n_samples)
        beta, se, t, p, ns = _ols_scan(synthetic_lrr, phenotype)
        assert beta.shape == (n_markers,)
        assert se.shape == (n_markers,)
        assert t.shape == (n_markers,)
        assert p.shape == (n_markers,)
        assert ns.shape == (n_markers,)

    def test_p_values_in_range(self, synthetic_lrr: np.ndarray) -> None:
        n_samples = synthetic_lrr.shape[1]
        phenotype = np.random.default_rng(42).normal(size=n_samples)
        _, _, _, p, _ = _ols_scan(synthetic_lrr, phenotype)
        assert np.all((p >= 0) & (p <= 1))

    def test_known_signal(self) -> None:
        """Strong linear signal should produce small p-value."""
        rng = np.random.default_rng(99)
        n_markers, n_samples = 10, 200
        lrr = rng.normal(size=(n_markers, n_samples))
        # Phenotype strongly correlated with marker 0
        phenotype = 2.0 * lrr[0] + rng.normal(0, 0.1, n_samples)
        beta, se, t, p, ns = _ols_scan(lrr, phenotype)
        assert p[0] < 1e-10
        assert abs(beta[0] - 2.0) < 0.2

    def test_no_signal(self) -> None:
        """Independent LRR and phenotype should not yield tiny p-values."""
        rng = np.random.default_rng(77)
        n_markers, n_samples = 20, 100
        lrr = rng.normal(size=(n_markers, n_samples))
        phenotype = rng.normal(size=n_samples)
        _, _, _, p, _ = _ols_scan(lrr, phenotype)
        assert np.median(p) > 0.05

    def test_with_covariates(self) -> None:
        rng = np.random.default_rng(55)
        n_markers, n_samples = 5, 100
        lrr = rng.normal(size=(n_markers, n_samples))
        cov = rng.normal(size=(n_samples, 2))
        phenotype = cov[:, 0] + rng.normal(0, 0.1, n_samples)
        beta, se, t, p, ns = _ols_scan(lrr, phenotype, covariates=cov)
        # With confounding accounted for, LRR markers should be non-sig
        assert np.all(ns == n_samples)
        assert beta.shape == (n_markers,)

    def test_handles_nan(self) -> None:
        """OLS scan should fall back to complete-case when NaN present."""
        rng = np.random.default_rng(33)
        n_markers, n_samples = 5, 50
        lrr = rng.normal(size=(n_markers, n_samples))
        lrr[0, :10] = np.nan
        phenotype = rng.normal(size=n_samples)
        beta, se, t, p, ns = _ols_scan(lrr, phenotype)
        assert ns[0] == 40
        assert ns[1] == 50
        assert np.all(p >= 0)

    def test_complete_vs_missing_paths(self) -> None:
        """Results should match between paths when no data is missing."""
        rng = np.random.default_rng(111)
        n_markers, n_samples = 5, 60
        lrr = rng.normal(size=(n_markers, n_samples))
        phenotype = rng.normal(size=n_samples)

        from array_lrr_gwas.association import (
            _ols_scan_complete,
            _ols_scan_missing,
        )

        C = np.ones((n_samples, 1))
        b1, s1, t1, p1, n1 = _ols_scan_complete(lrr, phenotype, C, 1)
        b2, s2, t2, p2, n2 = _ols_scan_missing(lrr, phenotype, C, 1)

        np.testing.assert_allclose(b1, b2, atol=1e-10)
        np.testing.assert_allclose(s1, s2, atol=1e-10)
        np.testing.assert_allclose(p1, p2, atol=1e-10)


# ---------------------------------------------------------------------------
# Logistic scan
# ---------------------------------------------------------------------------

class TestLogisticScan:
    """Tests for the per-marker logistic regression scan."""

    def test_output_shapes(self) -> None:
        rng = np.random.default_rng(42)
        n_markers, n_samples = 5, 100
        lrr = rng.normal(size=(n_markers, n_samples))
        phenotype = (rng.random(n_samples) > 0.5).astype(float)
        beta, se, z, p, ns = _logistic_scan(lrr, phenotype)
        assert beta.shape == (n_markers,)
        assert p.shape == (n_markers,)

    def test_known_signal_logistic(self) -> None:
        """Strong effect should produce small p-value."""
        rng = np.random.default_rng(88)
        n_markers, n_samples = 5, 500
        lrr = rng.normal(size=(n_markers, n_samples))
        prob = 1.0 / (1.0 + np.exp(-3.0 * lrr[0]))
        phenotype = (rng.random(n_samples) < prob).astype(float)
        beta, se, z, p, ns = _logistic_scan(lrr, phenotype)
        assert p[0] < 0.01
        assert beta[0] > 0

    def test_handles_nan_logistic(self) -> None:
        rng = np.random.default_rng(22)
        n_markers, n_samples = 3, 80
        lrr = rng.normal(size=(n_markers, n_samples))
        lrr[0, :20] = np.nan
        phenotype = (rng.random(n_samples) > 0.5).astype(float)
        beta, se, z, p, ns = _logistic_scan(lrr, phenotype)
        assert ns[0] == 60
        assert ns[1] == 80


# ---------------------------------------------------------------------------
# run_association (public API)
# ---------------------------------------------------------------------------

class TestRunAssociation:
    """Tests for the ``run_association`` entry point."""

    def test_ols_basic(self, synthetic_lrr: np.ndarray) -> None:
        n_markers, n_samples = synthetic_lrr.shape
        rng = np.random.default_rng(7)
        phenotype = rng.normal(size=n_samples)
        variants = _make_variants(n_markers)

        result = run_association(synthetic_lrr, phenotype, variants)
        assert isinstance(result, AssociationResult)
        assert result.method == "ols"
        assert len(result.chrom) == n_markers
        assert result.beta.shape == (n_markers,)
        assert np.all((result.p_value >= 0) & (result.p_value <= 1))

    def test_logistic_basic(self) -> None:
        rng = np.random.default_rng(8)
        n_markers, n_samples = 5, 100
        lrr = rng.normal(size=(n_markers, n_samples))
        phenotype = (rng.random(n_samples) > 0.5).astype(float)
        variants = _make_variants(n_markers)

        result = run_association(lrr, phenotype, variants, method="logistic")
        assert result.method == "logistic"
        assert result.beta.shape == (n_markers,)

    def test_with_covariates(self) -> None:
        rng = np.random.default_rng(9)
        n_markers, n_samples = 5, 100
        lrr = rng.normal(size=(n_markers, n_samples))
        phenotype = rng.normal(size=n_samples)
        cov = rng.normal(size=(n_samples, 3))
        variants = _make_variants(n_markers)

        result = run_association(lrr, phenotype, variants, covariates=cov)
        assert result.beta.shape == (n_markers,)

    def test_to_records(self) -> None:
        rng = np.random.default_rng(10)
        n_markers, n_samples = 3, 50
        lrr = rng.normal(size=(n_markers, n_samples))
        phenotype = rng.normal(size=n_samples)
        variants = _make_variants(n_markers)

        result = run_association(lrr, phenotype, variants)
        records = result.to_records()
        assert len(records) == n_markers
        assert "chrom" in records[0]
        assert "p_value" in records[0]

    def test_variant_id_fallback(self) -> None:
        """When variant has no 'id', chrom:pos is used."""
        rng = np.random.default_rng(11)
        n_markers, n_samples = 2, 30
        lrr = rng.normal(size=(n_markers, n_samples))
        phenotype = rng.normal(size=n_samples)
        variants = [{"chrom": "chr1", "pos": 100}, {"chrom": "chr2", "pos": 200}]

        result = run_association(lrr, phenotype, variants)
        assert result.variant_id == ["chr1:100", "chr2:200"]

    # --- Error handling ---

    def test_bad_phenotype_shape(self) -> None:
        lrr = np.zeros((3, 5))
        phenotype = np.zeros(10)  # wrong length
        with pytest.raises(ValueError, match="phenotype"):
            run_association(lrr, phenotype, _make_variants(3))

    def test_bad_variants_length(self) -> None:
        lrr = np.zeros((3, 5))
        phenotype = np.zeros(5)
        with pytest.raises(ValueError, match="len\\(variants\\)"):
            run_association(lrr, phenotype, _make_variants(2))

    def test_bad_covariates_shape(self) -> None:
        lrr = np.zeros((3, 5))
        phenotype = np.zeros(5)
        cov = np.zeros((10, 2))  # wrong n_samples
        with pytest.raises(ValueError, match="covariates"):
            run_association(lrr, phenotype, _make_variants(3), covariates=cov)

    def test_unknown_method(self) -> None:
        lrr = np.zeros((3, 5))
        phenotype = np.zeros(5)
        with pytest.raises(ValueError, match="Unknown method"):
            run_association(lrr, phenotype, _make_variants(3), method="glm")

    def test_logistic_non_binary_raises(self) -> None:
        lrr = np.zeros((3, 5))
        phenotype = np.array([0.0, 0.5, 1.0, 0.0, 1.0])
        with pytest.raises(ValueError, match="binary"):
            run_association(
                lrr, phenotype, _make_variants(3), method="logistic"
            )


# ---------------------------------------------------------------------------
# Integration: real BCF data
# ---------------------------------------------------------------------------

class TestAssociationWithBcf:
    """End-to-end tests using ``tests/data/test.bcf``."""

    def test_ols_on_test_bcf(self, test_bcf_path) -> None:
        """Read real BCF, run OLS association, verify output shape."""
        from array_lrr_gwas.io_vcf import read_lrr

        lrr, samples, variants = read_lrr(test_bcf_path)
        n_markers, n_samples = lrr.shape
        rng = np.random.default_rng(42)
        phenotype = rng.normal(size=n_samples)

        result = run_association(lrr, phenotype, variants)
        assert len(result.chrom) == n_markers
        assert result.beta.shape == (n_markers,)
        assert np.all(result.p_value >= 0)

    def test_logistic_on_test_bcf(self, test_bcf_path) -> None:
        """Read real BCF, run logistic association."""
        from array_lrr_gwas.io_vcf import read_lrr

        lrr, samples, variants = read_lrr(test_bcf_path)
        n_markers, n_samples = lrr.shape
        rng = np.random.default_rng(42)
        phenotype = (rng.random(n_samples) > 0.5).astype(float)

        result = run_association(lrr, phenotype, variants, method="logistic")
        assert result.method == "logistic"
        assert result.beta.shape == (n_markers,)

    def test_corrected_bcf_association(self, test_bcf_path) -> None:
        """Full pipeline: correct → associate."""
        from array_lrr_gwas.correction import correct_lrr
        from array_lrr_gwas.io_vcf import read_lrr

        lrr, samples, variants = read_lrr(test_bcf_path)
        n_markers, n_samples = lrr.shape
        positions = np.array([v["pos"] for v in variants], dtype=np.intp)
        chromosomes = np.array([v["chrom"] for v in variants], dtype=str)

        corrected, info = correct_lrr(
            lrr,
            positions=positions,
            chromosomes=chromosomes,
            max_lrr_sd=10.0,
            min_sample_call_rate=0.0,
            min_marker_call_rate=0.0,
            min_var=0.0,
            k=2,
        )
        rng = np.random.default_rng(42)
        phenotype = rng.normal(size=n_samples)

        result = run_association(corrected, phenotype, variants)
        assert result.beta.shape == (n_markers,)
        assert np.all(result.n_samples >= 0)
