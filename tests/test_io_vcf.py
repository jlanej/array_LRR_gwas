"""Tests for BCF/VCF I/O with corrected LRR values."""

from pathlib import Path

import numpy as np
import pytest

from array_lrr_gwas.io_vcf import read_lrr, write_corrected, _CORRECTION_HEADER_KEY
from array_lrr_gwas.correction import correct_lrr


class TestReadLrr:
    def test_read_test_bcf(self, test_bcf_path):
        lrr, samples, variants = read_lrr(test_bcf_path)
        assert lrr.shape == (100, 20)
        assert len(samples) == 20
        assert len(variants) == 100

    def test_variant_metadata(self, test_bcf_path):
        _, _, variants = read_lrr(test_bcf_path)
        v0 = variants[0]
        assert "chrom" in v0
        assert "pos" in v0
        assert "ref" in v0

    def test_samples_are_strings(self, test_bcf_path):
        _, samples, _ = read_lrr(test_bcf_path)
        assert all(isinstance(s, str) for s in samples)


class TestWriteCorrected:
    def test_roundtrip(self, test_bcf_path, tmp_path):
        lrr, samples, variants = read_lrr(test_bcf_path)
        # Write back unchanged values as VCF
        info = {
            "k": 2,
            "backend": "rsvd",
            "n_hq_samples": 17,
            "n_markers_used": 90,
            "singular_values": np.array([1.0, 0.5]),
        }
        out_vcf = tmp_path / "out.vcf"
        write_corrected(out_vcf, lrr, samples, variants, info)

        # Read back
        lrr2, samples2, variants2 = read_lrr(out_vcf)
        assert samples2 == samples
        assert lrr2.shape == lrr.shape
        # Non-NaN values should be close (float rounding)
        valid = ~np.isnan(lrr) & ~np.isnan(lrr2)
        np.testing.assert_allclose(lrr2[valid], lrr[valid], atol=1e-4)

    def test_bcf_output(self, test_bcf_path, tmp_path):
        lrr, samples, variants = read_lrr(test_bcf_path)
        info = {
            "k": 1,
            "backend": "rsvd",
            "n_hq_samples": 20,
            "n_markers_used": 100,
            "singular_values": np.array([1.0]),
        }
        out_bcf = tmp_path / "out.bcf"
        write_corrected(out_bcf, lrr, samples, variants, info)
        assert out_bcf.exists()

        lrr2, samples2, _ = read_lrr(out_bcf)
        assert lrr2.shape == lrr.shape

    def test_header_contains_correction_info(self, test_bcf_path, tmp_path):
        lrr, samples, variants = read_lrr(test_bcf_path)
        info = {
            "k": 3,
            "backend": "rsvd",
            "n_hq_samples": 17,
            "n_markers_used": 90,
            "singular_values": np.array([2.0, 1.0, 0.5]),
        }
        out_vcf = tmp_path / "out.vcf"
        write_corrected(out_vcf, lrr, samples, variants, info)

        import pysam

        vf = pysam.VariantFile(str(out_vcf))
        header_str = str(vf.header)
        vf.close()
        assert _CORRECTION_HEADER_KEY in header_str
        assert "Components removed (k): 3" in header_str


class TestEndToEndWithBcf:
    """Integration: read BCF → correct → write → read back."""

    def test_full_pipeline(self, test_bcf_path, tmp_path):
        lrr, samples, variants = read_lrr(test_bcf_path)

        corrected, info = correct_lrr(
            lrr,
            k=3,
            max_lrr_sd=10.0,
            min_sample_call_rate=0.0,
            min_marker_call_rate=0.5,
            min_var=0.0,
        )

        out_bcf = tmp_path / "corrected.bcf"
        write_corrected(
            out_bcf,
            corrected,
            samples,
            variants,
            info,
            path_template=test_bcf_path,
        )

        lrr2, samples2, _ = read_lrr(out_bcf)
        assert lrr2.shape == lrr.shape
        assert samples2 == samples

        # Correction should have reduced variance
        orig_var = np.nanvar(lrr)
        corr_var = np.nanvar(lrr2)
        assert corr_var < orig_var
