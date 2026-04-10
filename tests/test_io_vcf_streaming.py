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
        all_variants, n_skipped, _ = stream_correct_write(
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
        all_variants, n_skipped, _ = stream_correct_write(
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
        all_variants, n_skipped, _ = stream_correct_write(
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

    def test_no_diagnostic_ids_returns_none(self, test_bcf_path, tmp_path):
        """Without diagnostic_marker_ids, post_metrics is None."""
        from array_lrr_gwas.correction import correct_lrr

        lrr, samples, variants = read_lrr(test_bcf_path)
        _, info = correct_lrr(
            lrr, k=2, max_lrr_sd=10.0,
            min_sample_call_rate=0.0, min_marker_call_rate=0.5,
            min_var=0.0,
        )
        Vt_k = np.asarray(info["sample_scores"])[:2, :]

        out_path = tmp_path / "no_diag.bcf"
        _, _, post_metrics = stream_correct_write(
            test_bcf_path, out_path, Vt_k, samples, info,
            path_template=test_bcf_path,
        )
        assert post_metrics is None

    def test_diagnostic_metrics_match_in_memory(self, test_bcf_path, tmp_path):
        """Streaming diagnostic metrics match in-memory compute_sample_metrics."""
        from array_lrr_gwas.correction import correct_lrr, residualize_qr
        from array_lrr_gwas.interactive_report import compute_sample_metrics

        lrr, samples, variants = read_lrr(test_bcf_path)
        _, info = correct_lrr(
            lrr, k=2, max_lrr_sd=10.0,
            min_sample_call_rate=0.0, min_marker_call_rate=0.5,
            min_var=0.0,
        )
        Vt_k = np.asarray(info["sample_scores"])[:2, :]

        # Build variant IDs for a subset
        def _vid(v):
            vid = v.get("id")
            if vid is not None and vid != ".":
                return vid
            alts = v.get("alts") or ()
            return f"{v['chrom']}:{v['pos']}:{v.get('ref', '')}:{':'.join(alts)}"

        all_ids = [_vid(v) for v in variants]
        # Use every 10th marker as the diagnostic subset
        subset_indices = list(range(0, len(variants), 10))
        diagnostic_ids = {all_ids[i] for i in subset_indices}

        # Stream-correct with diagnostic tracking
        out_path = tmp_path / "diag_metrics.bcf"
        _, _, post_metrics = stream_correct_write(
            test_bcf_path, out_path, Vt_k, samples, info,
            path_template=test_bcf_path,
            diagnostic_marker_ids=diagnostic_ids,
        )

        assert post_metrics is not None
        assert set(post_metrics.keys()) == {"SAMPLE", "LRR_SD", "callrate", "n_markers_used"}
        assert post_metrics["SAMPLE"] == list(samples)
        assert len(post_metrics["LRR_SD"]) == len(samples)
        assert len(post_metrics["callrate"]) == len(samples)

        # Compute reference metrics in-memory on the same subset
        corrected_full = residualize_qr(lrr, Vt_k)
        corrected_subset = corrected_full[subset_indices]
        ref_metrics = compute_sample_metrics(corrected_subset, list(samples))

        # Compare LRR_SD values (should be very close)
        for i in range(len(samples)):
            ref_sd = ref_metrics["LRR_SD"][i]
            stream_sd = post_metrics["LRR_SD"][i]
            if ref_sd is None:
                assert stream_sd is None
            else:
                assert stream_sd == pytest.approx(ref_sd, abs=1e-6), (
                    f"LRR_SD mismatch for sample {samples[i]}: "
                    f"streaming={stream_sd}, in-memory={ref_sd}"
                )

        # Compare callrate values (should be identical)
        for i in range(len(samples)):
            assert post_metrics["callrate"][i] == pytest.approx(
                ref_metrics["callrate"][i], abs=1e-10,
            )

    def test_diagnostic_empty_subset(self, test_bcf_path, tmp_path):
        """Empty diagnostic set returns None post_metrics."""
        from array_lrr_gwas.correction import correct_lrr

        lrr, samples, variants = read_lrr(test_bcf_path)
        _, info = correct_lrr(
            lrr, k=2, max_lrr_sd=10.0,
            min_sample_call_rate=0.0, min_marker_call_rate=0.5,
            min_var=0.0,
        )
        Vt_k = np.asarray(info["sample_scores"])[:2, :]

        # No matching IDs
        out_path = tmp_path / "empty_diag.bcf"
        _, _, post_metrics = stream_correct_write(
            test_bcf_path, out_path, Vt_k, samples, info,
            path_template=test_bcf_path,
            diagnostic_marker_ids={"NONEXISTENT_ID"},
        )
        assert post_metrics is None
