"""Tests for LD pruning (array_lrr_gwas.ld_prune)."""

from __future__ import annotations

import numpy as np
import pytest

from array_lrr_gwas.ld_prune import _r2_vec, ld_prune


# ---------------------------------------------------------------------------
# _r2_vec
# ---------------------------------------------------------------------------

class TestR2Vec:
    """Tests for the vectorised r² helper."""

    def test_identical_vectors(self) -> None:
        x = np.array([1.0, -1.0, 0.5, -0.5])
        block = x.reshape(1, -1)
        r2 = _r2_vec(x, block)
        np.testing.assert_allclose(r2, [1.0], atol=1e-10)

    def test_uncorrelated(self) -> None:
        x = np.array([1.0, -1.0, 1.0, -1.0])
        y = np.array([1.0, 1.0, -1.0, -1.0])
        block = y.reshape(1, -1)
        r2 = _r2_vec(x, block)
        np.testing.assert_allclose(r2, [0.0], atol=1e-10)

    def test_zero_variance_x(self) -> None:
        x = np.zeros(5)
        block = np.ones((3, 5))
        r2 = _r2_vec(x, block)
        np.testing.assert_array_equal(r2, [0.0, 0.0, 0.0])

    def test_multiple_rows(self) -> None:
        rng = np.random.default_rng(99)
        x = rng.standard_normal(50)
        block = np.vstack([x, -x, rng.standard_normal(50)])
        r2 = _r2_vec(x, block)
        assert r2.shape == (3,)
        np.testing.assert_allclose(r2[0], 1.0, atol=1e-10)
        np.testing.assert_allclose(r2[1], 1.0, atol=1e-10)
        assert 0 <= r2[2] <= 1


# ---------------------------------------------------------------------------
# ld_prune – basic behaviour
# ---------------------------------------------------------------------------

class TestLdPrune:
    """Tests for the main ``ld_prune`` function."""

    def test_returns_bool_mask(self) -> None:
        rng = np.random.default_rng(1)
        dosage = rng.choice([0.0, 1.0, 2.0], size=(20, 30))
        mask = ld_prune(dosage)
        assert mask.dtype == bool
        assert mask.shape == (20,)

    def test_empty_input(self) -> None:
        dosage = np.empty((0, 10), dtype=float)
        mask = ld_prune(dosage)
        assert mask.shape == (0,)

    def test_all_independent_kept(self) -> None:
        """Truly independent variants should all survive pruning."""
        rng = np.random.default_rng(2)
        n_vars, n_samp = 50, 200
        dosage = rng.choice([0.0, 1.0, 2.0], size=(n_vars, n_samp))
        mask = ld_prune(dosage, r2_thresh=0.99)
        # With r²<0.99, essentially everything should be kept for random data
        assert mask.sum() == n_vars

    def test_perfect_duplicates_pruned(self) -> None:
        """Exact duplicates should be aggressively pruned."""
        rng = np.random.default_rng(3)
        base = rng.choice([0.0, 1.0, 2.0], size=(1, 100))
        # 10 identical variants
        dosage = np.tile(base, (10, 1))
        mask = ld_prune(dosage, r2_thresh=0.2)
        # Only the first should survive (all others are in perfect LD)
        assert mask[0]
        assert mask.sum() == 1

    def test_prune_reduces_count(self) -> None:
        """LD pruning should reduce variant count for correlated data."""
        rng = np.random.default_rng(4)
        n_samp = 100
        # Create blocks of correlated variants
        base1 = rng.choice([0.0, 1.0, 2.0], size=(1, n_samp))
        base2 = rng.choice([0.0, 1.0, 2.0], size=(1, n_samp))
        block1 = np.tile(base1, (20, 1))
        block2 = np.tile(base2, (20, 1))
        # Add small noise to avoid monomorphism
        block1 += rng.normal(0, 0.01, block1.shape)
        block2 += rng.normal(0, 0.01, block2.shape)
        dosage = np.vstack([block1, block2])
        mask = ld_prune(dosage, r2_thresh=0.2)
        assert mask.sum() < dosage.shape[0]

    def test_handles_nan(self) -> None:
        """Missing genotypes should be mean-imputed before pruning."""
        rng = np.random.default_rng(5)
        dosage = rng.choice([0.0, 1.0, 2.0], size=(20, 30))
        dosage[0, :5] = np.nan
        dosage[3, 10:15] = np.nan
        mask = ld_prune(dosage, r2_thresh=0.99)
        assert mask.dtype == bool
        assert mask.shape == (20,)

    def test_strict_threshold_prunes_more(self) -> None:
        """Lower r² threshold should prune more variants."""
        rng = np.random.default_rng(6)
        dosage = rng.choice([0.0, 1.0, 2.0], size=(100, 50))
        mask_loose = ld_prune(dosage, r2_thresh=0.8)
        mask_strict = ld_prune(dosage, r2_thresh=0.05)
        assert mask_strict.sum() <= mask_loose.sum()


# ---------------------------------------------------------------------------
# ld_prune – with genomic coordinates
# ---------------------------------------------------------------------------

class TestLdPruneWithCoords:
    """Tests for LD pruning with chromosome/position information."""

    def test_cross_chromosome_not_pruned(self) -> None:
        """Identical variants on different chromosomes must not prune each other."""
        base = np.array([[0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 0.0]])
        dosage = np.tile(base, (4, 1))
        chroms = np.array(["chr1", "chr1", "chr2", "chr2"])
        positions = np.array([100, 200, 100, 200])
        mask = ld_prune(
            dosage, positions=positions, chromosomes=chroms,
            r2_thresh=0.2, window_bp=10_000,
        )
        # First on each chromosome kept; second on each pruned
        expected = [True, False, True, False]
        np.testing.assert_array_equal(mask, expected)

    def test_distant_variants_not_pruned(self) -> None:
        """Variants beyond window_bp should not prune each other."""
        base = np.array([[0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 0.0]])
        dosage = np.tile(base, (2, 1))
        chroms = np.array(["chr1", "chr1"])
        positions = np.array([100, 2_000_100])
        mask = ld_prune(
            dosage, positions=positions, chromosomes=chroms,
            r2_thresh=0.2, window_bp=1_000_000,
        )
        # Both kept because they are >1 Mb apart
        assert mask.all()

    def test_close_variants_pruned(self) -> None:
        """Identical variants within window_bp should be pruned."""
        base = np.array([[0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 0.0]])
        dosage = np.tile(base, (2, 1))
        chroms = np.array(["chr1", "chr1"])
        positions = np.array([100, 500])
        mask = ld_prune(
            dosage, positions=positions, chromosomes=chroms,
            r2_thresh=0.2, window_bp=1_000_000,
        )
        assert mask[0]
        assert not mask[1]

    def test_window_boundary(self) -> None:
        """A variant at exactly window_bp distance is within the window."""
        base = np.array([[0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 0.0]])
        dosage = np.tile(base, (2, 1))
        chroms = np.array(["chr1", "chr1"])
        # Second variant at exactly window_bp
        positions = np.array([100, 100 + 500])
        mask = ld_prune(
            dosage, positions=positions, chromosomes=chroms,
            r2_thresh=0.2, window_bp=500,
        )
        # Within window → second should be pruned
        assert mask[0]
        assert not mask[1]


# ---------------------------------------------------------------------------
# ld_prune_plink2 availability check
# ---------------------------------------------------------------------------

class TestPlink2Backend:
    """Tests for the plink2 backend helper."""

    def test_plink2_not_found_raises(self) -> None:
        """FileNotFoundError when plink2 is not on PATH."""
        from array_lrr_gwas.ld_prune import _plink2_available

        # This test is informational; if plink2 IS installed it passes trivially
        if not _plink2_available():
            from array_lrr_gwas.ld_prune import ld_prune_plink2

            with pytest.raises(FileNotFoundError, match="plink2"):
                ld_prune_plink2("/nonexistent.bcf")
