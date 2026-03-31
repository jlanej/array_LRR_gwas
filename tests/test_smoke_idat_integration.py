"""Integration / smoke test: illumina_idat_processing → array-LRR-GWAS.

Mimics the end-to-end workflow of running ``array-lrr-gwas`` directly on
typical ``illumina_idat_processing`` output using the bundled test BCF and
synthetic fixture files laid out in the upstream directory structure.
"""

from __future__ import annotations

import csv
import os
import textwrap
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import BCF_N_SAMPLES, BCF_N_VARIANTS, BCF_SAMPLES, TEST_BCF

# ---------------------------------------------------------------------------
# Helpers — create fixture files that mimic upstream pipeline output
# ---------------------------------------------------------------------------

def _write_sample_sheet(path: Path, samples: list[str]) -> None:
    """Write a minimal ``compiled_sample_sheet.tsv``."""
    rng = np.random.default_rng(42)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        header = [
            "Sample_ID", "call_rate", "lrr_sd", "baf_sd",
            "sex_status", "inbreeding_F",
            "pre_pca_excluded", "excluded_relatedness",
            "excluded_het_outlier",
        ] + [f"PC{i}" for i in range(1, 21)]
        writer.writerow(header)
        for sid in samples:
            row = [
                sid,
                f"{rng.uniform(0.97, 1.0):.4f}",
                f"{rng.uniform(0.05, 0.20):.4f}",
                f"{rng.uniform(0.01, 0.10):.4f}",
                "CONCORDANT",
                f"{rng.uniform(-0.02, 0.02):.4f}",
                "false", "false", "false",
            ] + [f"{rng.normal():.6f}" for _ in range(20)]
            writer.writerow(row)


def _write_phenotype(path: Path, samples: list[str]) -> None:
    """Write a synthetic phenotype file."""
    rng = np.random.default_rng(99)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["sample_id", "phenotype"])
        for sid in samples:
            writer.writerow([sid, f"{rng.normal():.4f}"])


def _write_variant_qc(path: Path, bcf_path: Path) -> None:
    """Write a minimal ``collated_variant_qc.tsv`` for the test BCF."""
    from array_lrr_gwas.io_vcf import read_lrr

    _, _, variants = read_lrr(str(bcf_path))
    rng = np.random.default_rng(7)

    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow([
            "variant_id",
            "all_ancestries_call_rate_pass",
            "all_ancestries_hwe_pass",
            "all_ancestries_maf_pass",
        ])
        for v in variants:
            vid = v.get("id")
            if not vid or vid == ".":
                alts = v.get("alts") or ()
                vid = f"{v['chrom']}:{v['pos']}:{v.get('ref', '')}:{':'.join(alts)}"
            # Most variants pass; a few randomly fail
            cr = "True" if rng.random() > 0.02 else "False"
            hwe = "True" if rng.random() > 0.02 else "False"
            maf = "True" if rng.random() > 0.05 else "False"
            writer.writerow([vid, cr, hwe, maf])


def _setup_upstream_layout(
    base: Path,
    bcf_path: Path,
) -> dict[str, Path]:
    """Create a directory layout mirroring illumina_idat_processing output.

    Returns a dict of logical name → path for each fixture file.
    """
    stage2_vcf = base / "stage2" / "vcf"
    stage2_vcf.mkdir(parents=True, exist_ok=True)
    qc_dir = base / "ancestry_stratified_qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    # Symlink the real test BCF to the expected location
    link_bcf = stage2_vcf / "stage2_reclustered.bcf"
    link_csi = stage2_vcf / "stage2_reclustered.bcf.csi"
    if not link_bcf.exists():
        os.symlink(bcf_path, link_bcf)
    csi = bcf_path.with_suffix(".bcf.csi")
    if csi.exists() and not link_csi.exists():
        os.symlink(csi, link_csi)

    sheet_path = base / "compiled_sample_sheet.tsv"
    _write_sample_sheet(sheet_path, BCF_SAMPLES)

    pheno_path = base / "phenotype.tsv"
    _write_phenotype(pheno_path, BCF_SAMPLES)

    vqc_path = qc_dir / "collated_variant_qc.tsv"
    _write_variant_qc(vqc_path, bcf_path)

    return {
        "bcf": link_bcf,
        "sample_sheet": sheet_path,
        "phenotype": pheno_path,
        "variant_qc": vqc_path,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def upstream_layout(tmp_path_factory) -> dict[str, Path]:
    """Module-scoped fixture: upstream illumina_idat_processing layout."""
    base = tmp_path_factory.mktemp("upstream")
    return _setup_upstream_layout(base, TEST_BCF)


class TestSmokeCorrect:
    """Smoke test: ``array-lrr-gwas correct`` on upstream output."""

    def test_correct_runs(self, upstream_layout, tmp_path) -> None:
        from array_lrr_gwas.cli import main as cli_main

        out = tmp_path / "corrected.bcf"
        args = [
            "correct",
            str(upstream_layout["bcf"]),
            "-o", str(out),
            "--variant-qc", str(upstream_layout["variant_qc"]),
            "-v",
        ]
        rc = cli_main(args)
        assert rc == 0
        assert out.exists()
        assert out.stat().st_size > 0


class TestSmokeAssociate:
    """Smoke test: ``array-lrr-gwas associate`` on upstream output."""

    def test_associate_ols_without_genotype_bcf(self, upstream_layout, tmp_path) -> None:
        """Run OLS association *without* --genotype-bcf (defaults to input)."""
        from array_lrr_gwas.cli import main as cli_main

        out = tmp_path / "results_ols.tsv"
        args = [
            "associate",
            str(upstream_layout["bcf"]),
            "--phenotype", str(upstream_layout["phenotype"]),
            "--sample-sheet", str(upstream_layout["sample_sheet"]),
            "--variant-qc", str(upstream_layout["variant_qc"]),
            "--method", "ols",
            "-o", str(out),
            "-v",
        ]
        rc = cli_main(args)
        assert rc == 0
        assert out.exists()

        # Verify output structure
        with open(out) as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            cols = reader.fieldnames
            assert "chrom" in cols
            assert "p_value" in cols
            # QC provenance columns should be present
            assert "all_ancestries_call_rate_pass" in cols
            assert "all_ancestries_hwe_pass" in cols
            assert "all_ancestries_maf_pass" in cols
            assert "intensity_only" in cols
            assert "lrr_monomorphic" in cols
            rows = list(reader)
            assert len(rows) > 0

    def test_associate_with_genotype_bcf(self, upstream_layout, tmp_path) -> None:
        """Run OLS association *with* explicit --genotype-bcf."""
        from array_lrr_gwas.cli import main as cli_main

        out = tmp_path / "results_explicit_gt.tsv"
        args = [
            "associate",
            str(upstream_layout["bcf"]),
            "--phenotype", str(upstream_layout["phenotype"]),
            "--sample-sheet", str(upstream_layout["sample_sheet"]),
            "--genotype-bcf", str(upstream_layout["bcf"]),
            "--variant-qc", str(upstream_layout["variant_qc"]),
            "--method", "ols",
            "-o", str(out),
            "-v",
        ]
        rc = cli_main(args)
        assert rc == 0
        assert out.exists()


class TestSmokeSegment:
    """Smoke test: ``array-lrr-gwas segment`` on association output."""

    def test_segment_after_associate(self, upstream_layout, tmp_path) -> None:
        from array_lrr_gwas.cli import main as cli_main

        assoc_out = tmp_path / "results.tsv"
        seg_out = tmp_path / "regions.bed"

        # Step 1: associate (OLS, fast)
        rc = cli_main([
            "associate",
            str(upstream_layout["bcf"]),
            "--phenotype", str(upstream_layout["phenotype"]),
            "--sample-sheet", str(upstream_layout["sample_sheet"]),
            "--variant-qc", str(upstream_layout["variant_qc"]),
            "--method", "ols",
            "-o", str(assoc_out),
            "-v",
        ])
        assert rc == 0

        # Step 2: segment
        rc = cli_main([
            "segment",
            str(assoc_out),
            "-o", str(seg_out),
            "-v",
        ])
        assert rc == 0
        assert seg_out.exists()
        assert seg_out.stat().st_size > 0


class TestSmokeFullPipeline:
    """End-to-end: correct → associate → segment with upstream layout."""

    def test_full_pipeline(self, upstream_layout, tmp_path) -> None:
        from array_lrr_gwas.cli import main as cli_main

        corrected = tmp_path / "corrected.bcf"
        results = tmp_path / "results.tsv"
        regions = tmp_path / "regions.bed"

        # correct
        rc = cli_main([
            "correct",
            str(upstream_layout["bcf"]),
            "-o", str(corrected),
            "--variant-qc", str(upstream_layout["variant_qc"]),
            "-v",
        ])
        assert rc == 0

        # associate (OLS — fast, no GRM needed)
        rc = cli_main([
            "associate",
            str(corrected),
            "--phenotype", str(upstream_layout["phenotype"]),
            "--sample-sheet", str(upstream_layout["sample_sheet"]),
            "--variant-qc", str(upstream_layout["variant_qc"]),
            "--method", "ols",
            "-o", str(results),
            "-v",
        ])
        assert rc == 0

        # segment
        rc = cli_main([
            "segment",
            str(results),
            "-o", str(regions),
            "-v",
        ])
        assert rc == 0
        assert regions.exists()


class TestCaseInsensitiveColumns:
    """Verify sample_sheet parsing handles case variations for column names."""

    def test_read_sample_sheet_lowercase_sample_id(self, tmp_path) -> None:
        """``sample_id`` (lowercase) should work as ``Sample_ID``."""
        from array_lrr_gwas.sample_sheet import read_sample_sheet

        tsv = tmp_path / "sheet.tsv"
        tsv.write_text(
            "sample_id\tPC1\tPC2\n"
            "S1\t0.1\t0.2\n"
            "S2\t0.3\t0.4\n"
        )
        ids, covs, names = read_sample_sheet(tsv)
        assert ids == ["S1", "S2"]
        assert covs.shape == (2, 2)

    def test_classify_samples_lowercase_columns(self, tmp_path) -> None:
        """All-lowercase columns should be resolved correctly."""
        from array_lrr_gwas.sample_sheet import classify_samples_from_sheet

        tsv = tmp_path / "sheet.tsv"
        tsv.write_text(
            "sample_id\tcall_rate\tlrr_sd\n"
            "S1\t0.99\t0.10\n"
            "S2\t0.80\t0.10\n"
        )
        hq = classify_samples_from_sheet(tsv)
        assert hq == {"S1"}

    def test_classify_for_association_mixed_case(self, tmp_path) -> None:
        """Mixed-case column names should be resolved."""
        from array_lrr_gwas.sample_sheet import classify_samples_for_association

        tsv = tmp_path / "sheet.tsv"
        # Use 'sample_id' instead of 'Sample_ID'
        tsv.write_text(
            "sample_id\tcall_rate\tlrr_sd\tbaf_sd\n"
            "S1\t0.99\t0.10\t0.05\n"
            "S2\t0.99\t0.10\t0.05\n"
        )
        result = classify_samples_for_association(tsv)
        assert result.hq_ids == {"S1", "S2"}
        assert result.total == 2

    def test_exact_case_preferred(self, tmp_path) -> None:
        """Exact case match should be preferred over case-insensitive."""
        from array_lrr_gwas.sample_sheet import _resolve_column

        fields = ["sample_id", "Sample_ID", "PC1"]
        # When requesting "Sample_ID", the exact match should win
        assert _resolve_column(fields, "Sample_ID") == "Sample_ID"
        # When requesting "sample_id", the exact match should also win
        assert _resolve_column(fields, "sample_id") == "sample_id"

    def test_no_match_returns_none(self, tmp_path) -> None:
        from array_lrr_gwas.sample_sheet import _resolve_column

        fields = ["PC1", "PC2"]
        assert _resolve_column(fields, "Sample_ID") is None
