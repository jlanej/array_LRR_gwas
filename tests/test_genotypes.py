"""Tests for genotype extraction (array_lrr_gwas.genotypes)."""

from __future__ import annotations

import numpy as np
import pytest

from array_lrr_gwas.genotypes import read_genotypes
from tests.conftest import BCF_N_SAMPLES, BCF_SAMPLES


class TestReadGenotypes:
    """Tests for ``read_genotypes``."""

    def test_reads_gt_from_test_bcf(self, test_bcf_path) -> None:
        """Test BCF contains real GT data from illumina_idat_processing."""
        dosage, samples, variants = read_genotypes(
            test_bcf_path, min_maf=0.0, min_call_rate=0.0,
        )
        assert dosage.shape[0] > 0
        assert dosage.shape[1] == BCF_N_SAMPLES
        assert len(samples) == BCF_N_SAMPLES

    def test_returns_samples(self, test_bcf_path) -> None:
        dosage, samples, variants = read_genotypes(test_bcf_path)
        assert isinstance(samples, list)
        assert all(isinstance(s, str) for s in samples)
        assert samples == BCF_SAMPLES

    def test_maf_filter_reduces_variants(self, test_bcf_path) -> None:
        """Higher MAF threshold should yield fewer variants."""
        dosage_all, _, _ = read_genotypes(
            test_bcf_path, min_maf=0.0, min_call_rate=0.0,
        )
        dosage_filt, _, _ = read_genotypes(
            test_bcf_path, min_maf=0.05, min_call_rate=0.0,
        )
        assert dosage_filt.shape[0] < dosage_all.shape[0]
        assert dosage_filt.shape[0] > 0

    def test_dosage_values_in_range(self, test_bcf_path) -> None:
        """Dosage values should be 0, 1, 2, or NaN."""
        dosage, _, _ = read_genotypes(
            test_bcf_path, min_maf=0.0, min_call_rate=0.0,
        )
        valid = dosage[~np.isnan(dosage)]
        assert np.all(np.isin(valid, [0.0, 1.0, 2.0]))
