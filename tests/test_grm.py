"""Tests for GRM computation (array_lrr_gwas.grm)."""

from __future__ import annotations

import numpy as np
import pytest

from array_lrr_gwas.grm import compute_grm


class TestComputeGrm:
    """Tests for ``compute_grm``."""

    def test_output_shape(self) -> None:
        rng = np.random.default_rng(42)
        n_variants, n_samples = 100, 20
        dosage = rng.choice([0.0, 1.0, 2.0], size=(n_variants, n_samples))
        grm = compute_grm(dosage, min_maf=0.0)
        assert grm.shape == (n_samples, n_samples)

    def test_symmetric(self) -> None:
        rng = np.random.default_rng(43)
        dosage = rng.choice([0.0, 1.0, 2.0], size=(200, 30))
        grm = compute_grm(dosage, min_maf=0.0)
        np.testing.assert_allclose(grm, grm.T, atol=1e-10)

    def test_positive_semidefinite(self) -> None:
        rng = np.random.default_rng(44)
        dosage = rng.choice([0.0, 1.0, 2.0], size=(200, 30))
        grm = compute_grm(dosage, min_maf=0.0)
        eigenvalues = np.linalg.eigvalsh(grm)
        assert np.all(eigenvalues >= -1e-8)

    def test_diagonal_close_to_one(self) -> None:
        """For unrelated samples, diagonal should be ~1."""
        rng = np.random.default_rng(45)
        n_variants, n_samples = 5000, 20
        # Simulate independent genotypes with MAF=0.3
        dosage = rng.binomial(2, 0.3, size=(n_variants, n_samples)).astype(float)
        grm = compute_grm(dosage, min_maf=0.01)
        diag = np.diag(grm)
        np.testing.assert_allclose(diag, 1.0, atol=0.2)

    def test_maf_filter(self) -> None:
        rng = np.random.default_rng(46)
        n_variants, n_samples = 100, 50
        dosage = rng.choice([0.0, 1.0, 2.0], size=(n_variants, n_samples))
        # Make some variants monomorphic
        dosage[:10] = 0.0
        grm1 = compute_grm(dosage, min_maf=0.0)
        grm2 = compute_grm(dosage, min_maf=0.01)
        # Both should produce valid GRMs
        assert grm1.shape == (n_samples, n_samples)
        assert grm2.shape == (n_samples, n_samples)

    def test_handles_nan(self) -> None:
        """Missing genotypes should be mean-imputed."""
        rng = np.random.default_rng(47)
        dosage = rng.choice([0.0, 1.0, 2.0], size=(100, 20))
        dosage[0, :5] = np.nan
        grm = compute_grm(dosage, min_maf=0.0)
        assert grm.shape == (20, 20)
        assert not np.any(np.isnan(grm))

    def test_all_filtered_raises(self) -> None:
        """All monomorphic should raise ValueError."""
        dosage = np.zeros((10, 5))
        with pytest.raises(ValueError, match="No variants"):
            compute_grm(dosage, min_maf=0.01)
