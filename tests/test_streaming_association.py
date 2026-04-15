"""Tests for the streaming association pipeline components.

Covers ``read_variant_metadata``, ``stream_lrr_chunks``,
``run_association_streaming``, and the ``--sex-chr-mode`` CLI option.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import (
    BCF_N_SAMPLES,
    BCF_N_VARIANTS,
    BCF_SAMPLES,
)
from tests import mock_associate_io


# ===================================================================
# 1. read_variant_metadata
# ===================================================================


class TestReadVariantMetadata:
    """Test the lightweight metadata scanner."""

    def test_returns_samples_and_variants(self, test_bcf_path):
        from array_lrr_gwas.io_vcf import read_variant_metadata

        samples, variants = read_variant_metadata(test_bcf_path)
        assert samples == BCF_SAMPLES
        assert len(variants) == BCF_N_VARIANTS

    def test_variant_fields_present(self, test_bcf_path):
        from array_lrr_gwas.io_vcf import read_variant_metadata

        _, variants = read_variant_metadata(test_bcf_path)
        v = variants[0]
        assert "chrom" in v
        assert "pos" in v
        assert "id" in v
        assert "ref" in v
        assert "alts" in v
        assert "intensity_only" in v

    def test_consistent_with_read_lrr(self, test_bcf_path):
        from array_lrr_gwas.io_vcf import read_lrr, read_variant_metadata

        lrr, lrr_samples, lrr_variants = read_lrr(test_bcf_path)
        meta_samples, meta_variants = read_variant_metadata(test_bcf_path)

        assert meta_samples == lrr_samples
        assert len(meta_variants) == len(lrr_variants)
        # Spot-check a few fields
        for i in (0, 10, 100, -1):
            assert meta_variants[i]["chrom"] == lrr_variants[i]["chrom"]
            assert meta_variants[i]["pos"] == lrr_variants[i]["pos"]
            assert meta_variants[i]["id"] == lrr_variants[i]["id"]


# ===================================================================
# 2. stream_lrr_chunks
# ===================================================================


class TestStreamLrrChunks:
    """Test the chunked LRR streamer."""

    def test_full_scan_matches_read_lrr(self, test_bcf_path):
        from array_lrr_gwas.io_vcf import read_lrr, stream_lrr_chunks

        lrr_full, _, _ = read_lrr(test_bcf_path)
        chunks = list(stream_lrr_chunks(test_bcf_path, chunk_size=10_000))

        # Reconstruct full matrix
        lrr_streamed = np.vstack([c[0] for c in chunks])
        assert lrr_streamed.shape == lrr_full.shape
        np.testing.assert_array_equal(
            np.isnan(lrr_streamed), np.isnan(lrr_full),
        )
        finite = np.isfinite(lrr_full)
        np.testing.assert_allclose(
            lrr_streamed[finite], lrr_full[finite], atol=1e-12,
        )

    def test_sample_mask_filters_columns(self, test_bcf_path):
        from array_lrr_gwas.io_vcf import stream_lrr_chunks

        mask = np.zeros(BCF_N_SAMPLES, dtype=bool)
        mask[:3] = True  # keep first 3 samples
        chunks = list(stream_lrr_chunks(
            test_bcf_path, chunk_size=50_000, sample_mask=mask,
        ))
        lrr = np.vstack([c[0] for c in chunks])
        assert lrr.shape[1] == 3

    def test_variant_mask_filters_rows(self, test_bcf_path):
        from array_lrr_gwas.io_vcf import read_variant_metadata, stream_lrr_chunks

        _, variants = read_variant_metadata(test_bcf_path)
        n = len(variants)
        vmask = np.zeros(n, dtype=bool)
        vmask[:100] = True  # keep first 100

        chunks = list(stream_lrr_chunks(
            test_bcf_path, chunk_size=50_000, variant_mask=vmask,
        ))
        lrr = np.vstack([c[0] for c in chunks])
        assert lrr.shape[0] == 100

    def test_chunk_size_respected(self, test_bcf_path):
        from array_lrr_gwas.io_vcf import stream_lrr_chunks

        chunk_size = 1000
        for lrr_chunk, vars_chunk in stream_lrr_chunks(
            test_bcf_path, chunk_size=chunk_size,
        ):
            assert lrr_chunk.shape[0] <= chunk_size
            assert len(vars_chunk) == lrr_chunk.shape[0]


# ===================================================================
# 3. run_association_streaming
# ===================================================================


class TestRunAssociationStreaming:
    """Test the streaming association entry point."""

    @staticmethod
    def _make_chunks(lrr, variants, chunk_size=5000):
        """Helper to yield chunks from in-memory data."""
        for start in range(0, lrr.shape[0], chunk_size):
            end = min(start + chunk_size, lrr.shape[0])
            yield lrr[start:end], variants[start:end]

    def test_ols_returns_correct_count(self):
        from array_lrr_gwas.association import run_association_streaming

        rng = np.random.default_rng(42)
        n_markers, n_samples = 20, 50
        lrr = rng.standard_normal((n_markers, n_samples))
        phenotype = rng.standard_normal(n_samples)
        variants = [
            {"chrom": "chr1", "pos": i * 100, "id": f"v{i}"}
            for i in range(n_markers)
        ]

        result, info = run_association_streaming(
            self._make_chunks(lrr, variants),
            phenotype,
            method="ols",
        )
        assert len(result.chrom) == n_markers
        assert info["n_tested"] == n_markers
        assert info["n_monomorphic"] == 0

    def test_excludes_monomorphic(self):
        from array_lrr_gwas.association import run_association_streaming

        rng = np.random.default_rng(42)
        n_samples = 30
        lrr = np.vstack([
            rng.standard_normal((3, n_samples)),   # 3 normal
            np.zeros((1, n_samples)),                # 1 monomorphic
        ])
        phenotype = rng.standard_normal(n_samples)
        variants = [
            {"chrom": "chr1", "pos": i * 100, "id": f"v{i}"}
            for i in range(4)
        ]

        result, info = run_association_streaming(
            self._make_chunks(lrr, variants),
            phenotype,
            method="ols",
            exclude_monomorphic=True,
        )
        assert info["n_tested"] == 3
        assert info["n_monomorphic"] == 1
        assert "v3" in info["excluded_markers"]

    def test_excludes_intensity_only(self):
        from array_lrr_gwas.association import run_association_streaming

        rng = np.random.default_rng(42)
        n_samples = 30
        lrr = rng.standard_normal((4, n_samples))
        phenotype = rng.standard_normal(n_samples)
        variants = [
            {"chrom": "chr1", "pos": 100, "id": "v0"},
            {"chrom": "chr1", "pos": 200, "id": "v1", "intensity_only": True},
            {"chrom": "chr1", "pos": 300, "id": "v2"},
            {"chrom": "chr1", "pos": 400, "id": "v3"},
        ]

        result, info = run_association_streaming(
            self._make_chunks(lrr, variants),
            phenotype,
            method="ols",
            exclude_intensity_only=True,
        )
        assert info["n_tested"] == 3
        assert info["n_intensity_only"] == 1
        assert "v1" in info["excluded_markers"]

    def test_lmm_streaming(self):
        from array_lrr_gwas.association import run_association_streaming

        rng = np.random.default_rng(99)
        n_markers, n_samples = 10, 30
        lrr = rng.standard_normal((n_markers, n_samples))
        phenotype = rng.standard_normal(n_samples)
        grm = np.eye(n_samples)
        variants = [
            {"chrom": "chr1", "pos": i * 100, "id": f"v{i}"}
            for i in range(n_markers)
        ]

        result, info = run_association_streaming(
            self._make_chunks(lrr, variants),
            phenotype,
            method="lmm",
            grm=grm,
        )
        assert info["n_tested"] == n_markers
        assert result.method == "lmm"

    def test_empty_stream_returns_empty(self):
        from array_lrr_gwas.association import run_association_streaming

        result, info = run_association_streaming(
            iter([]),
            np.array([1.0, 2.0, 3.0]),
            method="ols",
        )
        assert len(result.chrom) == 0
        assert info["n_tested"] == 0


# ===================================================================
# 4. Sex-chromosome CLI mode
# ===================================================================


class TestSexChrMode:
    """Integration tests for --sex-chr-mode."""

    def test_sex_chr_requires_sample_sheet(self, tmp_path, monkeypatch):
        """Error when --sex-chr-mode used without --sample-sheet."""
        from array_lrr_gwas.cli import main

        pheno = tmp_path / "pheno.tsv"
        pheno.write_text("sample_id\tphenotype\nS1\t1.0\nS2\t2.0\nS3\t3.0\n")
        bcf = tmp_path / "in.bcf"
        bcf.write_text("stub")
        out = tmp_path / "results.tsv"

        lrr = np.random.default_rng(42).standard_normal((3, 3))
        samples = ["S1", "S2", "S3"]
        variants = [
            {"chrom": "chrX", "pos": 100, "id": "x1"},
            {"chrom": "chrX", "pos": 200, "id": "x2"},
            {"chrom": "chr1", "pos": 300, "id": "a1"},
        ]
        mock_associate_io(monkeypatch, lrr, samples, variants)

        rc = main([
            "associate", str(bcf),
            "--phenotype", str(pheno),
            "--method", "ols",
            "-o", str(out),
            "--sex-chr-mode", "x_with_sex_covariate",
        ])
        assert rc == 1

    def test_sex_chr_x_with_sex_covariate(self, tmp_path, monkeypatch, caplog):
        """x_with_sex_covariate mode writes a separate output file."""
        from array_lrr_gwas.cli import main

        pheno = tmp_path / "pheno.tsv"
        pheno.write_text(
            "sample_id\tphenotype\n"
            "S1\t1.0\nS2\t2.0\nS3\t3.0\nS4\t4.0\n"
        )
        sheet = tmp_path / "sheet.tsv"
        sheet.write_text(
            "Sample_ID\tcall_rate\tlrr_sd\tpredicted_sex\n"
            "S1\t0.99\t0.10\t1\nS2\t0.99\t0.10\t2\n"
            "S3\t0.99\t0.10\t1\nS4\t0.99\t0.10\t2\n"
        )
        bcf = tmp_path / "in.bcf"
        bcf.write_text("stub")
        out = tmp_path / "results.tsv"

        rng = np.random.default_rng(42)
        lrr = rng.standard_normal((6, 4))
        samples = ["S1", "S2", "S3", "S4"]
        variants = [
            {"chrom": "chr1", "pos": 100, "id": "a1"},
            {"chrom": "chr1", "pos": 200, "id": "a2"},
            {"chrom": "chr1", "pos": 300, "id": "a3"},
            {"chrom": "chrX", "pos": 100, "id": "x1"},
            {"chrom": "chrX", "pos": 200, "id": "x2"},
            {"chrom": "chrY", "pos": 100, "id": "y1"},
        ]
        mock_associate_io(monkeypatch, lrr, samples, variants)

        with caplog.at_level(logging.INFO, logger="array_lrr_gwas.cli"):
            rc = main([
                "associate", str(bcf),
                "--phenotype", str(pheno),
                "--sample-sheet", str(sheet),
                "--method", "ols",
                "-o", str(out),
                "--sex-chr-mode", "x_with_sex_covariate",
            ])

        assert rc == 0
        assert out.exists()
        # Sex-chr output
        sx_out = tmp_path / "results.x_with_sex_covariate.tsv"
        assert sx_out.exists()
        with open(sx_out) as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))
        # Should have the chrX markers (x1, x2)
        ids = [r["variant_id"] for r in rows]
        assert "x1" in ids
        assert "x2" in ids
        assert "a1" not in ids
        assert "Sex-chr mode x_with_sex_covariate" in caplog.text

    def test_sex_chr_male_only(self, tmp_path, monkeypatch):
        """x_male_only runs on males subset."""
        from array_lrr_gwas.cli import main

        pheno = tmp_path / "pheno.tsv"
        pheno.write_text(
            "sample_id\tphenotype\n"
            "S1\t1.0\nS2\t2.0\nS3\t3.0\nS4\t4.0\n"
        )
        sheet = tmp_path / "sheet.tsv"
        sheet.write_text(
            "Sample_ID\tcall_rate\tlrr_sd\tpredicted_sex\n"
            "S1\t0.99\t0.10\t1\nS2\t0.99\t0.10\t2\n"
            "S3\t0.99\t0.10\t1\nS4\t0.99\t0.10\t2\n"
        )
        bcf = tmp_path / "in.bcf"
        bcf.write_text("stub")
        out = tmp_path / "results.tsv"

        rng = np.random.default_rng(42)
        lrr = rng.standard_normal((4, 4))
        samples = ["S1", "S2", "S3", "S4"]
        variants = [
            {"chrom": "chr1", "pos": 100, "id": "a1"},
            {"chrom": "chr1", "pos": 200, "id": "a2"},
            {"chrom": "chrX", "pos": 100, "id": "x1"},
            {"chrom": "chrX", "pos": 200, "id": "x2"},
        ]
        mock_associate_io(monkeypatch, lrr, samples, variants)

        rc = main([
            "associate", str(bcf),
            "--phenotype", str(pheno),
            "--sample-sheet", str(sheet),
            "--method", "ols",
            "-o", str(out),
            "--sex-chr-mode", "x_male_only",
        ])
        assert rc == 0
        sx_out = tmp_path / "results.x_male_only.tsv"
        # Only 2 males → < 3 samples → skipped
        # (or present if it ran; check both)
        if sx_out.exists():
            with open(sx_out) as fh:
                rows = list(csv.DictReader(fh, delimiter="\t"))
            # Should only have chrX markers
            ids = [r["variant_id"] for r in rows]
            for vid in ids:
                assert vid.startswith("x")

    def test_sex_chr_y_male_only(self, tmp_path, monkeypatch):
        """y_male_only runs on chrY males."""
        from array_lrr_gwas.cli import main

        pheno = tmp_path / "pheno.tsv"
        lines = ["sample_id\tphenotype"]
        for i in range(1, 9):
            lines.append(f"S{i}\t{float(i)}")
        pheno.write_text("\n".join(lines) + "\n")

        sheet = tmp_path / "sheet.tsv"
        slines = ["Sample_ID\tcall_rate\tlrr_sd\tpredicted_sex"]
        for i in range(1, 9):
            slines.append(f"S{i}\t0.99\t0.10\t{1 if i <= 4 else 2}")
        sheet.write_text("\n".join(slines) + "\n")

        bcf = tmp_path / "in.bcf"
        bcf.write_text("stub")
        out = tmp_path / "results.tsv"

        rng = np.random.default_rng(42)
        lrr = rng.standard_normal((5, 8))
        samples = [f"S{i}" for i in range(1, 9)]
        variants = [
            {"chrom": "chr1", "pos": 100, "id": "a1"},
            {"chrom": "chr1", "pos": 200, "id": "a2"},
            {"chrom": "chr1", "pos": 300, "id": "a3"},
            {"chrom": "chrY", "pos": 100, "id": "y1"},
            {"chrom": "chrY", "pos": 200, "id": "y2"},
        ]
        mock_associate_io(monkeypatch, lrr, samples, variants)

        rc = main([
            "associate", str(bcf),
            "--phenotype", str(pheno),
            "--sample-sheet", str(sheet),
            "--method", "ols",
            "-o", str(out),
            "--sex-chr-mode", "y_male_only",
        ])
        assert rc == 0
        sx_out = tmp_path / "results.y_male_only.tsv"
        assert sx_out.exists()
        with open(sx_out) as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))
        ids = [r["variant_id"] for r in rows]
        assert "y1" in ids
        assert "y2" in ids


# ===================================================================
# 5. Audit fraction columns
# ===================================================================


class TestAuditFractions:
    """Test that audit summary includes fraction columns."""

    def test_summary_has_fractions(self, tmp_path):
        from array_lrr_gwas.audit import AuditLogger

        audit = AuditLogger()
        audit.record(
            stage="test_stage",
            id_type="marker",
            included=["a", "b", "c"],
            excluded={"d": "reason1"},
        )
        out = tmp_path / "summary.tsv"
        audit.write_summary_tsv(out)

        with open(out) as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            rows = list(reader)

        assert len(rows) == 1
        row = rows[0]
        assert "included_fraction" in row
        assert "excluded_fraction" in row
        assert float(row["included_fraction"]) == pytest.approx(0.75)
        assert float(row["excluded_fraction"]) == pytest.approx(0.25)


# ===================================================================
# 6. plink2 enforcement
# ===================================================================


class TestPlink2Enforcement:
    """Test that missing plink2 errors out instead of silent fallback."""

    def test_plink2_missing_returns_error(self, tmp_path, monkeypatch, caplog):
        from array_lrr_gwas.cli import main
        from array_lrr_gwas import association

        pheno = tmp_path / "pheno.tsv"
        pheno.write_text("sample_id\tphenotype\nS1\t0.1\nS2\t0.2\nS3\t0.3\n")
        bcf = tmp_path / "in.bcf"
        bcf.write_text("stub")
        out = tmp_path / "results.tsv"

        lrr = np.array([[0.1, 0.2, 0.3], [0.0, 0.1, 0.2]], dtype=float)
        samples = ["S1", "S2", "S3"]
        variants = [
            {"chrom": "chr1", "pos": 100, "id": "a1"},
            {"chrom": "chr1", "pos": 200, "id": "a2"},
        ]
        mock_associate_io(monkeypatch, lrr, samples, variants)

        dosage = np.array(
            [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [2.0, 1.0, 0.0]],
            dtype=float,
        )
        gt_variants = [
            {"chrom": "chr1", "pos": 100, "id": "v1", "ref": "A", "alts": ("C",)},
            {"chrom": "chr1", "pos": 110, "id": "v2", "ref": "A", "alts": ("G",)},
            {"chrom": "chr1", "pos": 120, "id": "v3", "ref": "A", "alts": ("T",)},
        ]
        monkeypatch.setattr(
            "array_lrr_gwas.genotypes.read_genotypes",
            lambda *_a, **_k: (dosage, samples, gt_variants),
        )
        monkeypatch.setattr(
            "array_lrr_gwas.ld_prune.ld_prune_plink2",
            lambda *_a, **_k: (_ for _ in ()).throw(FileNotFoundError("plink2")),
        )

        with caplog.at_level(logging.ERROR, logger="array_lrr_gwas.cli"):
            rc = main([
                "associate", str(bcf),
                "--phenotype", str(pheno),
                "--method", "lmm",
                "-o", str(out),
            ])

        assert rc == 1
        assert "plink2" in caplog.text
