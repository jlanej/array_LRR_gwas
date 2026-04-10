"""Tests for every-Nth marker selection."""

import pytest

from array_lrr_gwas.subsetting import select_every_nth


class TestSelectEveryNth:
    """Tests for select_every_nth()."""

    @pytest.fixture()
    def passing_ids(self):
        """A list of 100 variant IDs."""
        return [f"chr1:{i * 1000}:A:T" for i in range(100)]

    def test_target_ge_n_returns_all(self, passing_ids):
        """When target >= n, returns all IDs unchanged."""
        result = select_every_nth(passing_ids, target_n=200)
        assert result == passing_ids

    def test_target_eq_n_returns_all(self, passing_ids):
        """When target == n, returns all IDs unchanged."""
        result = select_every_nth(passing_ids, target_n=100)
        assert result == passing_ids

    def test_output_length_le_target(self, passing_ids):
        """Output length should be ≤ target_n."""
        result = select_every_nth(passing_ids, target_n=30)
        assert len(result) <= 30

    def test_output_length_close_to_target(self, passing_ids):
        """Output should be close to target_n."""
        target = 25
        result = select_every_nth(passing_ids, target_n=target)
        assert len(result) == target

    def test_deterministic(self, passing_ids):
        """Same inputs produce identical output."""
        r1 = select_every_nth(passing_ids, target_n=30)
        r2 = select_every_nth(passing_ids, target_n=30)
        assert r1 == r2

    def test_preserves_order(self, passing_ids):
        """Selected IDs maintain their relative order."""
        result = select_every_nth(passing_ids, target_n=20)
        indices = [passing_ids.index(vid) for vid in result]
        assert indices == sorted(indices)

    def test_output_subset_of_input(self, passing_ids):
        """All output IDs are from the input."""
        result = select_every_nth(passing_ids, target_n=30)
        assert all(vid in passing_ids for vid in result)

    def test_even_spacing(self, passing_ids):
        """Selected markers should be approximately evenly spaced."""
        result = select_every_nth(passing_ids, target_n=10)
        indices = [passing_ids.index(vid) for vid in result]
        gaps = [indices[i + 1] - indices[i] for i in range(len(indices) - 1)]
        # All gaps should be equal (every-Nth selection)
        assert len(set(gaps)) == 1

    def test_empty_input(self):
        """Empty input returns empty output."""
        result = select_every_nth([], target_n=10)
        assert result == []

    def test_target_zero(self, passing_ids):
        """Target of 0 returns empty list."""
        result = select_every_nth(passing_ids, target_n=0)
        assert result == []

    def test_target_one(self, passing_ids):
        """Target of 1 returns the first element."""
        result = select_every_nth(passing_ids, target_n=1)
        assert len(result) == 1
        assert result[0] == passing_ids[0]

    def test_small_list(self):
        """Works with a list smaller than target."""
        ids = ["a", "b", "c"]
        result = select_every_nth(ids, target_n=10)
        assert result == ids

    def test_returns_list(self, passing_ids):
        """Always returns a list, not a different sequence type."""
        result = select_every_nth(passing_ids, target_n=10)
        assert isinstance(result, list)
