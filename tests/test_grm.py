"""Tests for GRM computation (array_lrr_gwas.grm)."""

from __future__ import annotations

import numpy as np
import pytest

from array_lrr_gwas.grm import compute_grm, compute_x_grm


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


class TestComputeXGrm:
    """Tests for ``compute_x_grm`` (X-chromosome GRM)."""

    def _make_x_dosage(
        self,
        n_variants: int = 500,
        n_males: int = 10,
        n_females: int = 10,
        p: float = 0.3,
        seed: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate chrX dosage with correct 0/2 male coding."""
        rng = np.random.default_rng(seed)
        n_samples = n_males + n_females
        dosage = np.empty((n_variants, n_samples), dtype=np.float64)

        # Males: hemizygous → 0 or 2
        for j in range(n_males):
            dosage[:, j] = rng.choice([0.0, 2.0], size=n_variants, p=[1 - p, p])

        # Females: diploid → 0, 1, or 2
        for j in range(n_males, n_samples):
            dosage[:, j] = rng.binomial(2, p, size=n_variants).astype(float)

        is_male = np.array(
            [True] * n_males + [False] * n_females, dtype=bool,
        )
        return dosage, is_male

    def test_output_shape(self) -> None:
        dosage, is_male = self._make_x_dosage()
        grm = compute_x_grm(dosage, is_male, min_maf=0.0)
        assert grm.shape == (20, 20)

    def test_symmetric(self) -> None:
        dosage, is_male = self._make_x_dosage()
        grm = compute_x_grm(dosage, is_male, min_maf=0.0)
        np.testing.assert_allclose(grm, grm.T, atol=1e-10)

    def test_positive_semidefinite(self) -> None:
        dosage, is_male = self._make_x_dosage(n_variants=2000)
        grm = compute_x_grm(dosage, is_male, min_maf=0.0)
        eigenvalues = np.linalg.eigvalsh(grm)
        assert np.all(eigenvalues >= -1e-8)

    def test_male_diagonal_approximately_two(self) -> None:
        """Male diagonal elements should be ~2 (hemizygous inbreeding)."""
        dosage, is_male = self._make_x_dosage(n_variants=5000, seed=101)
        grm = compute_x_grm(dosage, is_male, min_maf=0.01)
        male_diag = np.diag(grm)[is_male]
        # Males should have diagonal ≈ 2.0 due to complete homozygosity
        np.testing.assert_allclose(male_diag, 2.0, atol=0.4)

    def test_female_diagonal_approximately_one(self) -> None:
        """Female diagonal elements should be ~1 (diploid)."""
        dosage, is_male = self._make_x_dosage(n_variants=5000, seed=102)
        grm = compute_x_grm(dosage, is_male, min_maf=0.01)
        female_diag = np.diag(grm)[~is_male]
        np.testing.assert_allclose(female_diag, 1.0, atol=0.3)

    def test_par_exclusion(self) -> None:
        """Variants in PAR regions should be excluded."""
        n_variants = 100
        dosage, is_male = self._make_x_dosage(n_variants=n_variants)

        # Place first 10 variants in PAR1
        positions = [
            ("chrX", 500_000 + i * 100) for i in range(10)  # PAR1
        ] + [
            ("chrX", 5_000_000 + i * 100) for i in range(n_variants - 10)  # non-PAR
        ]
        par_regions = {"chrX": [(0, 2_781_479)]}  # PAR1 only

        grm_with_par = compute_x_grm(
            dosage, is_male,
            variant_positions=positions, par_regions=par_regions,
            min_maf=0.0,
        )
        # Should have excluded 10 variants
        grm_no_par = compute_x_grm(
            dosage[10:], is_male,
            min_maf=0.0,
        )
        np.testing.assert_allclose(grm_with_par, grm_no_par, atol=1e-10)

    def test_male_01_rescaling(self) -> None:
        """Male dosage coded as 0/1 should be rescaled to 0/2."""
        rng = np.random.default_rng(103)
        n_variants, n_males, n_females = 200, 10, 10
        n_samples = n_males + n_females

        # Create dosage with correct 0/2 male coding
        dosage_02 = np.empty((n_variants, n_samples), dtype=np.float64)
        for j in range(n_males):
            dosage_02[:, j] = rng.choice([0.0, 2.0], size=n_variants, p=[0.7, 0.3])
        for j in range(n_males, n_samples):
            dosage_02[:, j] = rng.binomial(2, 0.3, size=n_variants).astype(float)

        # Create dosage with incorrect 0/1 male coding
        dosage_01 = dosage_02.copy()
        dosage_01[:, :n_males] = dosage_02[:, :n_males] / 2.0

        is_male = np.array([True] * n_males + [False] * n_females, dtype=bool)

        grm_02 = compute_x_grm(dosage_02, is_male, min_maf=0.0)
        grm_01 = compute_x_grm(dosage_01, is_male, min_maf=0.0)
        # After rescaling, both should produce the same X-GRM
        np.testing.assert_allclose(grm_01, grm_02, atol=1e-10)

    def test_handles_nan(self) -> None:
        """Missing genotypes should be mean-imputed."""
        dosage, is_male = self._make_x_dosage()
        dosage[0, :5] = np.nan
        grm = compute_x_grm(dosage, is_male, min_maf=0.0)
        assert grm.shape == (20, 20)
        assert not np.any(np.isnan(grm))

    def test_maf_filter(self) -> None:
        dosage, is_male = self._make_x_dosage()
        dosage[:10] = 0.0
        grm = compute_x_grm(dosage, is_male, min_maf=0.01)
        assert grm.shape == (20, 20)

    def test_all_filtered_raises(self) -> None:
        dosage = np.zeros((10, 5))
        is_male = np.array([True, True, False, False, False])
        with pytest.raises(ValueError, match="No X-chromosome variants"):
            compute_x_grm(dosage, is_male, min_maf=0.01)

    def test_dimension_mismatch_raises(self) -> None:
        dosage = np.random.default_rng(99).choice(
            [0.0, 1.0, 2.0], size=(50, 10),
        )
        is_male = np.array([True, False, True])  # wrong length
        with pytest.raises(ValueError, match="is_male must have shape"):
            compute_x_grm(dosage, is_male)

    def test_all_par_excluded_raises(self) -> None:
        """If all variants are in PAR, should raise."""
        dosage, is_male = self._make_x_dosage(n_variants=5)
        positions = [("chrX", 100_000 + i) for i in range(5)]
        par_regions = {"chrX": [(0, 999_999_999)]}
        with pytest.raises(ValueError, match="No X-chromosome variants remain after PAR"):
            compute_x_grm(
                dosage, is_male,
                variant_positions=positions, par_regions=par_regions,
            )
