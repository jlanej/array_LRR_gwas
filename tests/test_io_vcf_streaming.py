"""Tests for memory-efficient BCF I/O functions."""

from pathlib import Path

import numpy as np
import pytest

from array_lrr_gwas.io_vcf import (
    read_bcf_sample_ids,
    read_lrr,
    read_lrr_selected,
    stream_correct_write,
    _variant_id_from_rec,
)


class TestReadBcfSampleIds:
    """Tests for read_bcf_sample_ids()."""

    def test_returns_sample_list(self, test_bcf_path):
        """Returns a list of sample IDs."""
        samples = read_bcf_sample_ids(test_bcf_path)
        assert isinstance(samples, list)
        assert len(samples) > 0
        assert all(isinstance(s, str) for s in samples)

    def test_matches_read_lrr(self, test_bcf_path):
        """Sample list matches read_lrr() output."""
        samples_fast = read_bcf_sample_ids(test_bcf_path)
        _, samples_full, _ = read_lrr(test_bcf_path)
        assert samples_fast == samples_full


class TestReadLrrSelected:
    """Tests for read_lrr_selected()."""

    def test_loads_subset(self, test_bcf_path):
        """Loads only selected variants."""
        # Get all variant IDs
        _, samples, variants = read_lrr(test_bcf_path)
        all_ids = []
        for v in variants:
            vid = v.get("id")
            if vid is not None and vid != ".":
                all_ids.append(vid)
            else:
                alts = v.get("alts") or ()
                all_ids.append(f"{v['chrom']}:{v['pos']}:{v.get('ref', '')}:{':'.join(alts)}")

        # Select a subset of IDs
        selected = set(all_ids[:10])
        lrr_sub, samples_sub, variants_sub = read_lrr_selected(
            test_bcf_path, selected,
        )
        assert lrr_sub.shape[0] <= 10
        assert lrr_sub.shape[0] > 0
        assert lrr_sub.shape[1] == len(samples)
        assert len(variants_sub) == lrr_sub.shape[0]
        assert samples_sub == samples

    def test_empty_selection_returns_empty(self, test_bcf_path):
        """Empty selection set returns empty matrix."""
        lrr, samples, variants = read_lrr_selected(
            test_bcf_path, set(),
        )
        assert lrr.shape[0] == 0
        assert len(variants) == 0

    def test_nonexistent_ids_ignored(self, test_bcf_path):
        """IDs not in the BCF are silently skipped."""
        lrr, _, variants = read_lrr_selected(
            test_bcf_path, {"BOGUS_ID_123", "ANOTHER_FAKE"},
        )
        assert lrr.shape[0] == 0
        assert len(variants) == 0

    def test_values_match_full_read(self, test_bcf_path):
        """LRR values from selective read match full read."""
        lrr_full, _, variants_full = read_lrr(test_bcf_path)
        all_ids = []
        for v in variants_full:
            vid = v.get("id")
            if vid is not None and vid != ".":
                all_ids.append(vid)
            else:
                alts = v.get("alts") or ()
                all_ids.append(f"{v['chrom']}:{v['pos']}:{v.get('ref', '')}:{':'.join(alts)}")

        # Select first 5 IDs
        target_ids = all_ids[:5]
        selected = set(target_ids)
        lrr_sub, _, variants_sub = read_lrr_selected(test_bcf_path, selected)

        # Build a mapping from variant ID → row index in the subset
        sub_ids = []
        for v in variants_sub:
            vid = v.get("id")
            if vid is not None and vid != ".":
                sub_ids.append(vid)
            else:
                alts = v.get("alts") or ()
                sub_ids.append(f"{v['chrom']}:{v['pos']}:{v.get('ref', '')}:{':'.join(alts)}")

        # Verify each selected variant's LRR values match the full read
        for sub_row, sub_vid in enumerate(sub_ids):
            full_row = all_ids.index(sub_vid)
            np.testing.assert_array_equal(
                lrr_sub[sub_row],
                lrr_full[full_row],
                err_msg=f"LRR mismatch for variant {sub_vid}",
            )


class TestStreamCorrectWrite:
    """Tests for stream_correct_write()."""

    def test_basic_streaming(self, test_bcf_path, tmp_path):
        """Stream-correct a BCF and verify output."""
        from array_lrr_gwas.correction import correct_lrr

        # First do a full correction to get PC scores
        lrr, samples, variants = read_lrr(test_bcf_path)
        corrected, info = correct_lrr(
            lrr,
            k=2,
            max_lrr_sd=10.0,
            min_sample_call_rate=0.0,
            min_marker_call_rate=0.5,
            min_var=0.0,
        )
        k = info["k"]
        Vt_k = np.asarray(info["sample_scores"])[:k, :]

        # Now stream-correct
        out_path = tmp_path / "streamed.bcf"
        all_variants, n_skipped = stream_correct_write(
            test_bcf_path,
            out_path,
            Vt_k,
            samples,
            info,
            path_template=test_bcf_path,
        )

        # Verify
        assert len(all_variants) == len(variants)
        assert out_path.exists()

        # Read back and verify shape
        lrr_out, samples_out, _ = read_lrr(out_path)
        assert lrr_out.shape == lrr.shape
        assert samples_out == samples

    def test_streaming_vcf_output(self, test_bcf_path, tmp_path):
        """Stream-correct to VCF format."""
        from array_lrr_gwas.correction import correct_lrr

        lrr, samples, variants = read_lrr(test_bcf_path)
        corrected, info = correct_lrr(
            lrr,
            k=2,
            max_lrr_sd=10.0,
            min_sample_call_rate=0.0,
            min_marker_call_rate=0.5,
            min_var=0.0,
        )
        k = info["k"]
        Vt_k = np.asarray(info["sample_scores"])[:k, :]

        out_path = tmp_path / "streamed.vcf"
        all_variants, n_skipped = stream_correct_write(
            test_bcf_path,
            out_path,
            Vt_k,
            samples,
            info,
            path_template=test_bcf_path,
        )
        assert out_path.exists()
        assert len(all_variants) == len(variants)

    def test_streaming_with_intensity_only_markers(self, tmp_path):
        """stream_correct_write handles intensity-only markers (no ALT allele).

        Intensity-only probes from illumina_idat_processing have no ALT
        allele.  pysam requires at least 2 alleles when creating a new
        record, so the code must add a '.' placeholder ALT.  This test
        verifies the fix for the ValueError crash.
        """
        import pysam
        from array_lrr_gwas.correction import correct_lrr

        rng = np.random.default_rng(42)
        n_samples = 6
        sample_names = [f"S{i}" for i in range(n_samples)]

        # Build a BCF with a mix of regular and intensity-only markers
        bcf_path = tmp_path / "with_io.bcf"
        hdr = pysam.VariantHeader()
        hdr.add_meta(
            "FORMAT",
            items=[
                ("ID", "LRR"), ("Number", "1"),
                ("Type", "Float"), ("Description", "Log R Ratio"),
            ],
        )
        hdr.add_meta(
            "FORMAT",
            items=[
                ("ID", "GT"), ("Number", "1"),
                ("Type", "String"), ("Description", "Genotype"),
            ],
        )
        hdr.add_meta(
            "INFO",
            items=[
                ("ID", "INTENSITY_ONLY"), ("Number", "0"),
                ("Type", "Flag"),
                ("Description", "Intensity-only probe"),
            ],
        )
        hdr.add_meta("contig", items=[("ID", "chr1")])
        for s in sample_names:
            hdr.add_sample(s)

        vcf_out = pysam.VariantFile(str(bcf_path), "wb", header=hdr)

        n_regular = 30
        n_io = 20
        total = n_regular + n_io

        for i in range(n_regular):
            rec = vcf_out.new_record(
                contig="chr1",
                start=1000 * (i + 1),
                stop=1000 * (i + 1) + 1,
                alleles=("A", "C"),
                id=f"rs{i}",
            )
            for s in sample_names:
                rec.samples[s]["LRR"] = float(rng.normal(0, 0.1))
            vcf_out.write(rec)

        for i in range(n_io):
            # Intensity-only: REF only, no ALT (use "." placeholder)
            rec = vcf_out.new_record(
                contig="chr1",
                start=1000 * (n_regular + i + 1),
                stop=1000 * (n_regular + i + 1) + 1,
                alleles=("G", "."),
                id=f"io{i}",
            )
            rec.info["INTENSITY_ONLY"] = True
            for s in sample_names:
                rec.samples[s]["LRR"] = float(rng.normal(0, 0.05))
            vcf_out.write(rec)

        vcf_out.close()

        # Read and correct
        lrr, samples, variants = read_lrr(bcf_path)
        assert lrr.shape == (total, n_samples)

        corrected, info = correct_lrr(
            lrr,
            k=2,
            max_lrr_sd=10.0,
            min_sample_call_rate=0.0,
            min_marker_call_rate=0.5,
            min_var=0.0,
        )
        k = info["k"]
        Vt_k = np.asarray(info["sample_scores"])[:k, :]

        # Stream-correct (this used to crash with ValueError:
        # must set at least 2 alleles)
        out_path = tmp_path / "io_corrected.bcf"
        all_variants, n_skipped = stream_correct_write(
            bcf_path,
            out_path,
            Vt_k,
            samples,
            info,
            path_template=bcf_path,
        )

        assert len(all_variants) == total
        assert out_path.exists()

        # Verify output is readable and has all markers
        lrr_out, samples_out, variants_out = read_lrr(out_path)
        assert lrr_out.shape == (total, n_samples)
        assert samples_out == samples

        # Verify intensity-only markers are flagged in metadata
        io_count = sum(
            1 for v in all_variants if v.get("intensity_only", False)
        )
        assert io_count == n_io
