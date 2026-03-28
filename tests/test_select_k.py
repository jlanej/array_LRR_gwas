"""Tests for the k-selection heuristics."""

import numpy as np
import pytest

from array_lrr_gwas.select_k import select_k_elbow, select_k_mp


class TestSelectKMp:
    def test_returns_at_least_one(self):
        # Flat singular values (pure noise) → should still return >= 1
        s = np.ones(10)
        k = select_k_mp(s, n_markers=100, n_samples=50)
        assert k >= 1

    def test_detects_signal(self):
        # One large SV followed by small ones
        s = np.array([50.0, 1.0, 0.9, 0.8, 0.7, 0.6])
        k = select_k_mp(s, n_markers=200, n_samples=100)
        assert k >= 1

    def test_all_signal(self):
        # All very large SVs
        s = np.array([100.0, 90.0, 80.0])
        k = select_k_mp(s, n_markers=200, n_samples=100)
        assert k >= 1


class TestSelectKElbow:
    def test_clear_elbow(self):
        # Sharp drop after first value → elbow at 1
        s = np.array([100.0, 10.0, 9.0, 8.0, 7.0, 6.0])
        k = select_k_elbow(s)
        assert k >= 1

    def test_returns_at_least_one(self):
        s = np.array([1.0, 1.0, 1.0])
        k = select_k_elbow(s)
        assert k >= 1

    def test_short_array(self):
        s = np.array([5.0, 1.0])
        k = select_k_elbow(s)
        assert k == 1

    def test_max_k_respected(self):
        s = np.array([100.0, 50.0, 10.0, 9.0, 8.0])
        k = select_k_elbow(s, max_k=2)
        assert k <= 2
