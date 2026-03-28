"""Shared fixtures for array_lrr_gwas tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_BCF = TEST_DATA_DIR / "test.bcf"


@pytest.fixture
def test_bcf_path() -> Path:
    """Path to the bundled test BCF file."""
    assert TEST_BCF.exists(), f"Test BCF not found: {TEST_BCF}"
    return TEST_BCF


@pytest.fixture
def synthetic_lrr() -> np.ndarray:
    """A small synthetic LRR matrix with known properties.

    Shape: (50 markers, 15 samples).
    - 3 batch-effect components injected.
    - Last 2 samples have elevated noise (LQ).
    - Marker 0 has 50% missingness.
    - Marker 1 is constant (zero variance).
    """
    rng = np.random.default_rng(12345)
    n_markers, n_samples = 50, 15

    signal = rng.normal(0, 0.05, (n_markers, n_samples))
    U = rng.normal(0, 1, (n_markers, 3))
    V = rng.normal(0, 1, (3, n_samples))
    batch = 0.3 * U @ V / np.sqrt(n_markers)
    noise = rng.normal(0, 0.02, (n_markers, n_samples))
    lrr = signal + batch + noise

    # LQ samples
    lrr[:, -2:] += rng.normal(0, 0.5, (n_markers, 2))

    # Missingness
    lrr[0, :7] = np.nan

    # Zero variance
    lrr[1, :] = 0.0

    return lrr
