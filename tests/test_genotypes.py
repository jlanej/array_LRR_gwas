"""Tests for genotype extraction (array_lrr_gwas.genotypes)."""

from __future__ import annotations

import numpy as np
import pytest

from array_lrr_gwas.genotypes import read_genotypes


class TestReadGenotypes:
    """Tests for ``read_genotypes``."""

    def test_returns_empty_when_no_gt(self, test_bcf_path) -> None:
        """Test BCF has GT header but no actual GT values."""
        dosage, samples, variants = read_genotypes(
            test_bcf_path, min_maf=0.0, min_call_rate=0.0,
        )
        assert dosage.shape[0] == 0
        assert len(samples) == 20

    def test_returns_samples(self, test_bcf_path) -> None:
        dosage, samples, variants = read_genotypes(test_bcf_path)
        assert isinstance(samples, list)
        assert all(isinstance(s, str) for s in samples)

    def test_maf_filter_strict(self, test_bcf_path) -> None:
        """With strict MAF, no variants from empty GT file."""
        dosage, _, _ = read_genotypes(test_bcf_path, min_maf=0.05)
        assert dosage.shape[0] == 0
