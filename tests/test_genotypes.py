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

    def test_region_restricts_to_contig(self, test_bcf_path) -> None:
        """``region`` should restrict parsing to the requested contig via the index."""
        _, _, variants_all = read_genotypes(
            test_bcf_path, min_maf=0.0, min_call_rate=0.0, progress=False,
        )
        _, _, variants_x = read_genotypes(
            test_bcf_path,
            min_maf=0.0,
            min_call_rate=0.0,
            region="chrX",
            progress=False,
        )
        # chrX-only fetch must contain only chrX variants.
        assert len(variants_x) > 0
        assert all(v["chrom"] == "chrX" for v in variants_x)
        # And match the count from a full-file parse filtered to chrX.
        n_chrx_all = sum(1 for v in variants_all if v["chrom"] == "chrX")
        assert len(variants_x) == n_chrx_all

    def test_region_invalid_raises(self, test_bcf_path) -> None:
        """An unknown contig name should raise ValueError."""
        with pytest.raises(ValueError):
            read_genotypes(
                test_bcf_path,
                region="not_a_real_contig",
                progress=False,
            )
