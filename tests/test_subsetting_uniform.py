"""Tests for genome-uniform marker subsampling."""

import numpy as np
import pytest

from array_lrr_gwas.subsetting import subsample_markers_uniform


class TestSubsampleMarkersUniform:
    """Tests for subsample_markers_uniform()."""

    @pytest.fixture()
    def genome_data(self):
        """Synthetic genome with 3 chromosomes and 300 markers."""
        n_total = 300
        # 100 markers per chromosome
        chroms = np.array(
            ["chr1"] * 100 + ["chr2"] * 100 + ["chr3"] * 100
        )
        # Positions spread across each chromosome
        pos = np.concatenate([
            np.linspace(1_000_000, 250_000_000, 100, dtype=np.intp),
            np.linspace(1_000_000, 240_000_000, 100, dtype=np.intp),
            np.linspace(1_000_000, 200_000_000, 100, dtype=np.intp),
        ])
        # All markers are candidates
        candidates = np.arange(n_total, dtype=np.intp)
        return candidates, chroms, pos

    def test_no_op_when_target_ge_candidates(self, genome_data):
        """When target_n >= n_candidates, returns all candidates unchanged."""
        candidates, chroms, pos = genome_data
        result = subsample_markers_uniform(
            candidates, chroms, pos, target_n=500
        )
        np.testing.assert_array_equal(result, np.sort(candidates))

    def test_no_op_exact_match(self, genome_data):
        """When target_n == n_candidates, returns all."""
        candidates, chroms, pos = genome_data
        result = subsample_markers_uniform(
            candidates, chroms, pos, target_n=len(candidates)
        )
        np.testing.assert_array_equal(result, np.sort(candidates))

    def test_output_length_le_target(self, genome_data):
        """Output length should be ≤ target_n."""
        candidates, chroms, pos = genome_data
        result = subsample_markers_uniform(
            candidates, chroms, pos, target_n=50
        )
        assert len(result) <= 50

    def test_output_length_close_to_target(self, genome_data):
        """Output should be close to target_n (within 10%)."""
        candidates, chroms, pos = genome_data
        target = 100
        result = subsample_markers_uniform(
            candidates, chroms, pos, target_n=target
        )
        assert len(result) >= target * 0.9
        assert len(result) <= target

    def test_reproducible_same_seed(self, genome_data):
        """Same seed produces identical output."""
        candidates, chroms, pos = genome_data
        r1 = subsample_markers_uniform(
            candidates, chroms, pos, target_n=50, random_state=42
        )
        r2 = subsample_markers_uniform(
            candidates, chroms, pos, target_n=50, random_state=42
        )
        np.testing.assert_array_equal(r1, r2)

    def test_different_seed_different_output(self, genome_data):
        """Different seeds produce different outputs."""
        candidates, chroms, pos = genome_data
        # Use a target small enough that real selection happens within bins
        r1 = subsample_markers_uniform(
            candidates, chroms, pos, target_n=100, random_state=0
        )
        r2 = subsample_markers_uniform(
            candidates, chroms, pos, target_n=100, random_state=99
        )
        # With different seeds, outputs should differ
        assert not np.array_equal(r1, r2)

    def test_every_chromosome_represented(self, genome_data):
        """When target_n >= n_chroms, every chromosome is represented."""
        candidates, chroms, pos = genome_data
        result = subsample_markers_uniform(
            candidates, chroms, pos, target_n=30
        )
        result_chroms = chroms[result]
        assert set(result_chroms) == {"chr1", "chr2", "chr3"}

    def test_proportional_representation(self, genome_data):
        """Fraction per chromosome in output ≈ fraction in input (within 20%)."""
        candidates, chroms, pos = genome_data
        target = 150
        result = subsample_markers_uniform(
            candidates, chroms, pos, target_n=target
        )
        result_chroms = chroms[result]
        for chrom in ["chr1", "chr2", "chr3"]:
            frac_in = np.sum(chroms[candidates] == chrom) / len(candidates)
            frac_out = np.sum(result_chroms == chrom) / len(result)
            assert abs(frac_out - frac_in) < 0.20, (
                f"Chromosome {chrom}: input frac={frac_in:.2f}, "
                f"output frac={frac_out:.2f}"
            )

    def test_output_sorted(self, genome_data):
        """Output indices should be sorted."""
        candidates, chroms, pos = genome_data
        result = subsample_markers_uniform(
            candidates, chroms, pos, target_n=50
        )
        assert np.all(np.diff(result) >= 0)

    def test_output_subset_of_candidates(self, genome_data):
        """All output indices must be from the candidate set."""
        candidates, chroms, pos = genome_data
        result = subsample_markers_uniform(
            candidates, chroms, pos, target_n=50
        )
        assert np.all(np.isin(result, candidates))

    def test_partial_candidates(self):
        """Works when candidates are a subset of all markers."""
        n_total = 200
        chroms = np.array(["chr1"] * 100 + ["chr2"] * 100)
        pos = np.concatenate([
            np.linspace(1_000_000, 250_000_000, 100, dtype=np.intp),
            np.linspace(1_000_000, 240_000_000, 100, dtype=np.intp),
        ])
        # Only even-indexed markers are candidates
        candidates = np.arange(0, n_total, 2, dtype=np.intp)
        result = subsample_markers_uniform(
            candidates, chroms, pos, target_n=30
        )
        assert len(result) <= 30
        assert np.all(np.isin(result, candidates))

    def test_single_chromosome(self):
        """Works with a single chromosome."""
        chroms = np.array(["chr1"] * 100)
        pos = np.linspace(1_000_000, 250_000_000, 100, dtype=np.intp)
        candidates = np.arange(100, dtype=np.intp)
        result = subsample_markers_uniform(
            candidates, chroms, pos, target_n=20
        )
        assert len(result) <= 20
        assert len(result) > 0

    def test_very_small_target(self, genome_data):
        """Works with target_n=1."""
        candidates, chroms, pos = genome_data
        result = subsample_markers_uniform(
            candidates, chroms, pos, target_n=1
        )
        assert len(result) >= 1
        assert np.all(np.isin(result, candidates))
