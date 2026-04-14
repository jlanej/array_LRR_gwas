"""Shared fixtures for array_lrr_gwas tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_BCF = TEST_DATA_DIR / "test.bcf"

# ---- Constants describing the bundled test BCF (output of illumina_idat_processing) ----
BCF_N_VARIANTS = 96_869
BCF_N_SAMPLES = 8
BCF_SAMPLES = [
    "HG00268", "HG00513", "HG00731", "NA12878",
    "NA19129", "NA19238", "NA19331", "NA19347",
]
BCF_DETECTED_BUILD = "T2T-CHM13"


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


def mock_associate_io(monkeypatch, lrr, samples, variants):
    """Patch ``read_variant_metadata`` and ``stream_lrr_chunks`` for tests.

    Replaces the BCF I/O in ``_run_associate`` with in-memory data so that
    tests can run without a real BCF file.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
    lrr : ndarray, shape (n_variants, n_samples)
    samples : list of str
    variants : list of dict
    """
    monkeypatch.setattr(
        "array_lrr_gwas.io_vcf.read_variant_metadata",
        lambda _p: (list(samples), list(variants)),
    )

    def _fake_stream(path, *, chunk_size=5000, sample_mask=None, variant_mask=None):
        _lrr = lrr.copy()
        if sample_mask is not None:
            _lrr = _lrr[:, sample_mask]
        if variant_mask is not None:
            _vars = [v for v, m in zip(variants, variant_mask) if m]
            _lrr = _lrr[variant_mask]
        else:
            _vars = list(variants)
        for start in range(0, _lrr.shape[0], chunk_size):
            end = min(start + chunk_size, _lrr.shape[0])
            yield _lrr[start:end], _vars[start:end]

    monkeypatch.setattr(
        "array_lrr_gwas.io_vcf.stream_lrr_chunks",
        _fake_stream,
    )
