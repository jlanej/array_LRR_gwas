"""Tests for marker exclusion at correction and association stages."""

import csv
import logging

import numpy as np
import pytest

from array_lrr_gwas.cli import main
from array_lrr_gwas.qc_config import defaults, load_config


# ---------------------------------------------------------------------------
# INTENSITY_ONLY flag reading
# ---------------------------------------------------------------------------


class TestIntensityOnlyFlag:
    """Tests for INTENSITY_ONLY flag reading from BCF."""

    def test_intensity_only_read_from_bcf(self, test_bcf_path):
        """read_lrr populates 'intensity_only' key in variant metadata."""
        from array_lrr_gwas.io_vcf import read_lrr

        _, _, variants = read_lrr(test_bcf_path)
        assert len(variants) > 0
        for v in variants:
            assert "intensity_only" in v
            assert isinstance(v["intensity_only"], bool)

    def test_intensity_only_defaults_false_when_info_field_undefined(self, tmp_path):
        """When the INFO field is not defined, default to False."""
        import pysam

        out = tmp_path / "no_flag.vcf"
        hdr = pysam.VariantHeader()
        hdr.add_meta("contig", items=[("ID", "chr1")])
        hdr.add_meta(
            "FORMAT",
            items=[
                ("ID", "LRR"), ("Number", "1"), ("Type", "Float"),
                ("Description", "Log R Ratio"),
            ],
        )
        hdr.add_meta(
            "FORMAT",
            items=[
                ("ID", "GT"), ("Number", "1"), ("Type", "String"),
                ("Description", "Genotype"),
            ],
        )
        hdr.add_sample("S1")
        vcf_out = pysam.VariantFile(str(out), "w", header=hdr)
        rec = vcf_out.new_record(
            contig="chr1", start=99, stop=100, alleles=("A", "C"),
        )
        rec.samples["S1"]["LRR"] = 0.1
        vcf_out.write(rec)
        vcf_out.close()

        from array_lrr_gwas.io_vcf import read_lrr

        _, _, variants = read_lrr(out)
        assert len(variants) == 1
        assert variants[0]["intensity_only"] is False


# ---------------------------------------------------------------------------
# Subsetting per-step logging
# ---------------------------------------------------------------------------


class TestSubsettingLogging:
    """Verify that subset_markers emits per-step log messages."""

    def test_subset_markers_logs_per_step(self, caplog):
        """subset_markers logs per-filter counts."""
        from array_lrr_gwas.subsetting import subset_markers

        rng = np.random.default_rng(42)
        lrr = rng.standard_normal((20, 10))
        # Make first marker constant → excluded by variance
        lrr[0, :] = 0.5
        # Make second marker all-NaN → excluded by call rate
        lrr[1, :] = np.nan

        chroms = np.array(["chr1"] * 18 + ["chrX", "chrY"])
        pos = np.arange(20) * 1000

        with caplog.at_level(logging.INFO, logger="array_lrr_gwas.subsetting"):
            mask = subset_markers(
                lrr, positions=pos, chromosomes=chroms,
            )

        assert "call-rate filter" in caplog.text
        assert "variance filter" in caplog.text
        assert "autosome filter" in caplog.text
        assert "markers pass all filters" in caplog.text

    def test_subset_markers_logs_upstream_qc(self, caplog):
        """When upstream_qc_mask is provided, log its stats."""
        from array_lrr_gwas.subsetting import subset_markers

        rng = np.random.default_rng(42)
        lrr = rng.standard_normal((10, 5))
        qc_mask = np.ones(10, dtype=bool)
        qc_mask[0] = False

        with caplog.at_level(logging.INFO, logger="array_lrr_gwas.subsetting"):
            subset_markers(lrr, upstream_qc_mask=qc_mask)

        assert "upstream variant QC mask" in caplog.text


# ---------------------------------------------------------------------------
# Correction-stage logging
# ---------------------------------------------------------------------------


class TestCorrectionLogging:
    """Verify correct_lrr logs sample and marker stats."""

    def test_correct_lrr_logs_classification(self, synthetic_lrr, caplog):
        """correct_lrr logs HQ/LQ sample classification."""
        from array_lrr_gwas.correction import correct_lrr

        with caplog.at_level(logging.INFO, logger="array_lrr_gwas.correction"):
            correct_lrr(synthetic_lrr, k=1)

        assert "Sample classification" in caplog.text
        assert "HQ" in caplog.text


# ---------------------------------------------------------------------------
# qc_config association_marker_qc section
# ---------------------------------------------------------------------------


class TestAssociationMarkerQcConfig:
    """Test that association_marker_qc config section works correctly."""

    def test_defaults_include_association_marker_qc(self):
        cfg = defaults()
        assert "association_marker_qc" in cfg
        amqc = cfg["association_marker_qc"]
        assert amqc["exclude_intensity_only"] is True
        assert amqc["exclude_monomorphic_lrr"] is True
        assert "apply_variant_qc" not in amqc

    def test_yaml_override(self, tmp_path):
        cfg_file = tmp_path / "qc.yaml"
        cfg_file.write_text(
            "association_marker_qc:\n"
            "  exclude_intensity_only: false\n"
        )
        cfg = load_config(cfg_file)
        assert cfg["association_marker_qc"]["exclude_intensity_only"] is False
        # Unset keys keep defaults
        assert cfg["association_marker_qc"]["exclude_monomorphic_lrr"] is True


# ---------------------------------------------------------------------------
# CLI marker exclusion flags parsing
# ---------------------------------------------------------------------------


class TestMarkerExclusionArgParsing:
    """Test that CLI marker exclusion flags are parsed correctly."""

    def test_no_exclude_intensity_only_default(self):
        from array_lrr_gwas.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "associate", "in.bcf", "--phenotype", "p.tsv", "-o", "out.tsv",
        ])
        assert args.no_exclude_intensity_only is False

    def test_no_exclude_intensity_only_flag(self):
        from array_lrr_gwas.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "associate", "in.bcf", "--phenotype", "p.tsv", "-o", "out.tsv",
            "--no-exclude-intensity-only",
        ])
        assert args.no_exclude_intensity_only is True

    def test_no_exclude_monomorphic_lrr_flag(self):
        from array_lrr_gwas.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "associate", "in.bcf", "--phenotype", "p.tsv", "-o", "out.tsv",
            "--no-exclude-monomorphic-lrr",
        ])
        assert args.no_exclude_monomorphic_lrr is True


# ---------------------------------------------------------------------------
# Association-stage marker exclusion (integration)
# ---------------------------------------------------------------------------


class TestAssociationMarkerExclusion:
    """Integration tests for marker exclusion in the associate sub-command."""

    def test_intensity_only_excluded(self, tmp_path, monkeypatch, caplog):
        """Markers with intensity_only=True are excluded by default."""
        from array_lrr_gwas import association

        pheno = tmp_path / "pheno.tsv"
        pheno.write_text("sample_id\tphenotype\nS1\t0.1\nS2\t0.2\nS3\t0.3\n")
        out = tmp_path / "results.tsv"
        fake_bcf = tmp_path / "in.bcf"
        fake_bcf.write_text("stub")

        lrr = np.array([[0.1, 0.2, 0.3], [0.0, 0.1, 0.2], [0.3, 0.1, 0.2]], dtype=float)
        samples = ["S1", "S2", "S3"]
        variants = [
            {"chrom": "chr1", "pos": 100, "id": "a1", "intensity_only": False},
            {"chrom": "chr1", "pos": 200, "id": "a2", "intensity_only": True},
            {"chrom": "chr1", "pos": 300, "id": "a3", "intensity_only": False},
        ]
        monkeypatch.setattr(
            "array_lrr_gwas.io_vcf.read_lrr",
            lambda _p: (lrr, samples, variants),
        )

        class _FakeResult:
            chrom = ["chr1", "chr1"]

            @staticmethod
            def to_records():
                return [
                    {"chrom": "chr1", "pos": 100, "variant_id": "a1",
                     "beta": 0.0, "se": 1.0, "stat": 0.0, "p_value": 1.0,
                     "n_samples": 3, "method": "ols"},
                    {"chrom": "chr1", "pos": 300, "variant_id": "a3",
                     "beta": 0.0, "se": 1.0, "stat": 0.0, "p_value": 1.0,
                     "n_samples": 3, "method": "ols"},
                ]

        def _check_association(lrr_sub, phenotype, variants, **_k):
            # Only 2 markers should be passed (a1 and a3, not a2)
            assert lrr_sub.shape[0] == 2
            assert len(variants) == 2
            assert variants[0]["id"] == "a1"
            assert variants[1]["id"] == "a3"
            return _FakeResult()

        monkeypatch.setattr(association, "run_association", _check_association)

        with caplog.at_level(logging.INFO, logger="array_lrr_gwas.cli"):
            rc = main([
                "associate",
                str(fake_bcf),
                "--phenotype", str(pheno),
                "--method", "ols",
                "-o", str(out),
            ])

        assert rc == 0
        assert "INTENSITY_ONLY: 1 / 3 excluded" in caplog.text

    def test_intensity_only_retained_when_disabled(self, tmp_path, monkeypatch):
        """With --no-exclude-intensity-only, INTENSITY_ONLY markers are kept."""
        from array_lrr_gwas import association

        pheno = tmp_path / "pheno.tsv"
        pheno.write_text("sample_id\tphenotype\nS1\t0.1\nS2\t0.2\nS3\t0.3\n")
        out = tmp_path / "results.tsv"
        fake_bcf = tmp_path / "in.bcf"
        fake_bcf.write_text("stub")

        lrr = np.array([[0.1, 0.2, 0.3], [0.0, 0.1, 0.2]], dtype=float)
        samples = ["S1", "S2", "S3"]
        variants = [
            {"chrom": "chr1", "pos": 100, "id": "a1", "intensity_only": False},
            {"chrom": "chr1", "pos": 200, "id": "a2", "intensity_only": True},
        ]
        monkeypatch.setattr(
            "array_lrr_gwas.io_vcf.read_lrr",
            lambda _p: (lrr, samples, variants),
        )

        class _FakeResult:
            chrom = ["chr1", "chr1"]

            @staticmethod
            def to_records():
                return [
                    {"chrom": "chr1", "pos": 100, "variant_id": "a1",
                     "beta": 0.0, "se": 1.0, "stat": 0.0, "p_value": 1.0,
                     "n_samples": 3, "method": "ols"},
                    {"chrom": "chr1", "pos": 200, "variant_id": "a2",
                     "beta": 0.0, "se": 1.0, "stat": 0.0, "p_value": 1.0,
                     "n_samples": 3, "method": "ols"},
                ]

        def _check_all_kept(lrr_sub, phenotype, variants, **_k):
            assert lrr_sub.shape[0] == 2  # both kept
            return _FakeResult()

        monkeypatch.setattr(association, "run_association", _check_all_kept)

        rc = main([
            "associate",
            str(fake_bcf),
            "--phenotype", str(pheno),
            "--method", "ols",
            "--no-exclude-intensity-only",
            "-o", str(out),
        ])
        assert rc == 0

    def test_variant_qc_flags_propagated_not_filtered(self, tmp_path, monkeypatch, caplog):
        """Markers failing QC are still tested; QC flags propagated to output."""
        import csv

        from array_lrr_gwas import association

        # a1 passes all; a2 fails MAF; a3 fails call rate + HWE
        qc_tsv = tmp_path / "collated_variant_qc.tsv"
        qc_tsv.write_text(
            "variant_id\tall_ancestries_call_rate_pass\t"
            "all_ancestries_hwe_pass\tall_ancestries_maf_pass\n"
            "a1\tTrue\tTrue\tTrue\n"
            "a2\tTrue\tTrue\tFalse\n"
            "a3\tFalse\tFalse\tTrue\n"
        )

        pheno = tmp_path / "pheno.tsv"
        pheno.write_text("sample_id\tphenotype\nS1\t0.1\nS2\t0.2\nS3\t0.3\n")
        out = tmp_path / "results.tsv"
        fake_bcf = tmp_path / "in.bcf"
        fake_bcf.write_text("stub")

        lrr = np.array([[0.1, 0.2, 0.3], [0.0, 0.1, 0.2], [0.3, 0.1, 0.2]], dtype=float)
        samples = ["S1", "S2", "S3"]
        variants = [
            {"chrom": "chr1", "pos": 100, "id": "a1"},
            {"chrom": "chr1", "pos": 200, "id": "a2"},
            {"chrom": "chr1", "pos": 300, "id": "a3"},
        ]
        monkeypatch.setattr(
            "array_lrr_gwas.io_vcf.read_lrr",
            lambda _p: (lrr, samples, variants),
        )

        class _FakeResult:
            chrom = ["chr1", "chr1", "chr1"]

            @staticmethod
            def to_records():
                return [
                    {"chrom": "chr1", "pos": 100, "variant_id": "a1",
                     "beta": 0.0, "se": 1.0, "stat": 0.0, "p_value": 1.0,
                     "n_samples": 3, "method": "ols"},
                    {"chrom": "chr1", "pos": 200, "variant_id": "a2",
                     "beta": 0.0, "se": 1.0, "stat": 0.0, "p_value": 1.0,
                     "n_samples": 3, "method": "ols"},
                    {"chrom": "chr1", "pos": 300, "variant_id": "a3",
                     "beta": 0.0, "se": 1.0, "stat": 0.0, "p_value": 1.0,
                     "n_samples": 3, "method": "ols"},
                ]

        def _check(lrr_sub, phenotype, variants, **_k):
            # ALL 3 markers should be tested (no pre-filtering by QC)
            assert lrr_sub.shape[0] == 3
            assert len(variants) == 3
            return _FakeResult()

        monkeypatch.setattr(association, "run_association", _check)

        with caplog.at_level(logging.INFO, logger="array_lrr_gwas.cli"):
            rc = main([
                "associate",
                str(fake_bcf),
                "--phenotype", str(pheno),
                "--method", "ols",
                "--variant-qc", str(qc_tsv),
                "-o", str(out),
            ])

        assert rc == 0
        assert out.exists()

        # Verify QC provenance columns are present and correct
        with open(out, newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            rows = list(reader)

        assert len(rows) == 3
        # a1 passes all
        assert rows[0]["all_ancestries_call_rate_pass"] == "True"
        assert rows[0]["all_ancestries_hwe_pass"] == "True"
        assert rows[0]["all_ancestries_maf_pass"] == "True"
        # a2 fails MAF
        assert rows[1]["all_ancestries_call_rate_pass"] == "True"
        assert rows[1]["all_ancestries_hwe_pass"] == "True"
        assert rows[1]["all_ancestries_maf_pass"] == "False"
        # a3 fails call rate + HWE
        assert rows[2]["all_ancestries_call_rate_pass"] == "False"
        assert rows[2]["all_ancestries_hwe_pass"] == "False"
        assert rows[2]["all_ancestries_maf_pass"] == "True"

        # Verify marker-exclusion provenance columns are present
        for row in rows:
            assert "intensity_only" in row
            assert "lrr_monomorphic" in row
        # All markers in this test have non-zero variance and are not
        # INTENSITY_ONLY (variant dicts lack the key → defaults to False)
        for row in rows:
            assert row["intensity_only"] == "False"
            assert row["lrr_monomorphic"] == "False"

    def test_monomorphic_lrr_excluded(self, tmp_path, monkeypatch, caplog):
        """Markers with zero LRR variance are excluded by default."""
        from array_lrr_gwas import association

        pheno = tmp_path / "pheno.tsv"
        pheno.write_text("sample_id\tphenotype\nS1\t0.1\nS2\t0.2\nS3\t0.3\n")
        out = tmp_path / "results.tsv"
        fake_bcf = tmp_path / "in.bcf"
        fake_bcf.write_text("stub")

        # marker a1 has variation; a2 is constant (monomorphic)
        lrr = np.array([[0.1, 0.2, 0.3], [0.5, 0.5, 0.5]], dtype=float)
        samples = ["S1", "S2", "S3"]
        variants = [
            {"chrom": "chr1", "pos": 100, "id": "a1"},
            {"chrom": "chr1", "pos": 200, "id": "a2"},
        ]
        monkeypatch.setattr(
            "array_lrr_gwas.io_vcf.read_lrr",
            lambda _p: (lrr, samples, variants),
        )

        class _FakeResult:
            chrom = ["chr1"]

            @staticmethod
            def to_records():
                return [{
                    "chrom": "chr1", "pos": 100, "variant_id": "a1",
                    "beta": 0.0, "se": 1.0, "stat": 0.0, "p_value": 1.0,
                    "n_samples": 3, "method": "ols",
                }]

        def _check(lrr_sub, phenotype, variants, **_k):
            assert lrr_sub.shape[0] == 1  # only a1
            assert variants[0]["id"] == "a1"
            return _FakeResult()

        monkeypatch.setattr(association, "run_association", _check)

        with caplog.at_level(logging.INFO, logger="array_lrr_gwas.cli"):
            rc = main([
                "associate",
                str(fake_bcf),
                "--phenotype", str(pheno),
                "--method", "ols",
                "-o", str(out),
            ])

        assert rc == 0
        assert "monomorphic LRR" in caplog.text

    def test_monomorphic_retained_when_disabled(self, tmp_path, monkeypatch):
        """--no-exclude-monomorphic-lrr retains constant-LRR markers."""
        from array_lrr_gwas import association

        pheno = tmp_path / "pheno.tsv"
        pheno.write_text("sample_id\tphenotype\nS1\t0.1\nS2\t0.2\nS3\t0.3\n")
        out = tmp_path / "results.tsv"
        fake_bcf = tmp_path / "in.bcf"
        fake_bcf.write_text("stub")

        lrr = np.array([[0.1, 0.2, 0.3], [0.5, 0.5, 0.5]], dtype=float)
        samples = ["S1", "S2", "S3"]
        variants = [
            {"chrom": "chr1", "pos": 100, "id": "a1"},
            {"chrom": "chr1", "pos": 200, "id": "a2"},
        ]
        monkeypatch.setattr(
            "array_lrr_gwas.io_vcf.read_lrr",
            lambda _p: (lrr, samples, variants),
        )

        class _FakeResult:
            chrom = ["chr1", "chr1"]

            @staticmethod
            def to_records():
                return [
                    {"chrom": "chr1", "pos": 100, "variant_id": "a1",
                     "beta": 0.0, "se": 1.0, "stat": 0.0, "p_value": 1.0,
                     "n_samples": 3, "method": "ols"},
                    {"chrom": "chr1", "pos": 200, "variant_id": "a2",
                     "beta": 0.0, "se": 1.0, "stat": 0.0, "p_value": 1.0,
                     "n_samples": 3, "method": "ols"},
                ]

        def _check(lrr_sub, phenotype, variants, **_k):
            assert lrr_sub.shape[0] == 2  # both kept
            return _FakeResult()

        monkeypatch.setattr(association, "run_association", _check)

        rc = main([
            "associate",
            str(fake_bcf),
            "--phenotype", str(pheno),
            "--method", "ols",
            "--no-exclude-monomorphic-lrr",
            "-o", str(out),
        ])
        assert rc == 0

    def test_all_markers_excluded_returns_error(self, tmp_path, monkeypatch):
        """When all markers fail filters, return code 1."""
        pheno = tmp_path / "pheno.tsv"
        pheno.write_text("sample_id\tphenotype\nS1\t0.1\nS2\t0.2\nS3\t0.3\n")
        out = tmp_path / "results.tsv"
        fake_bcf = tmp_path / "in.bcf"
        fake_bcf.write_text("stub")

        # All markers are monomorphic
        lrr = np.array([[0.5, 0.5, 0.5], [0.3, 0.3, 0.3]], dtype=float)
        samples = ["S1", "S2", "S3"]
        variants = [
            {"chrom": "chr1", "pos": 100, "id": "a1"},
            {"chrom": "chr1", "pos": 200, "id": "a2"},
        ]
        monkeypatch.setattr(
            "array_lrr_gwas.io_vcf.read_lrr",
            lambda _p: (lrr, samples, variants),
        )

        rc = main([
            "associate",
            str(fake_bcf),
            "--phenotype", str(pheno),
            "--method", "ols",
            "-o", str(out),
        ])
        assert rc == 1

    def test_exclusion_summary_logged(self, tmp_path, monkeypatch, caplog):
        """Marker exclusion summary is logged with counts."""
        from array_lrr_gwas import association

        pheno = tmp_path / "pheno.tsv"
        pheno.write_text("sample_id\tphenotype\nS1\t0.1\nS2\t0.2\nS3\t0.3\n")
        out = tmp_path / "results.tsv"
        fake_bcf = tmp_path / "in.bcf"
        fake_bcf.write_text("stub")

        lrr = np.array([
            [0.1, 0.2, 0.3],  # varies
            [0.5, 0.5, 0.5],  # monomorphic
            [0.0, 0.4, 0.1],  # varies
        ], dtype=float)
        samples = ["S1", "S2", "S3"]
        variants = [
            {"chrom": "chr1", "pos": 100, "id": "a1"},
            {"chrom": "chr1", "pos": 200, "id": "a2"},
            {"chrom": "chr1", "pos": 300, "id": "a3"},
        ]
        monkeypatch.setattr(
            "array_lrr_gwas.io_vcf.read_lrr",
            lambda _p: (lrr, samples, variants),
        )

        class _FakeResult:
            chrom = ["chr1", "chr1"]

            @staticmethod
            def to_records():
                return [
                    {"chrom": "chr1", "pos": 100, "variant_id": "a1",
                     "beta": 0.0, "se": 1.0, "stat": 0.0, "p_value": 1.0,
                     "n_samples": 3, "method": "ols"},
                    {"chrom": "chr1", "pos": 300, "variant_id": "a3",
                     "beta": 0.0, "se": 1.0, "stat": 0.0, "p_value": 1.0,
                     "n_samples": 3, "method": "ols"},
                ]

        monkeypatch.setattr(
            association, "run_association",
            lambda *_a, **_k: _FakeResult(),
        )

        with caplog.at_level(logging.INFO, logger="array_lrr_gwas.cli"):
            rc = main([
                "associate",
                str(fake_bcf),
                "--phenotype", str(pheno),
                "--method", "ols",
                "-o", str(out),
            ])

        assert rc == 0
        assert "Association marker exclusion summary" in caplog.text
        assert "2 / 3 markers pass all filters" in caplog.text

    def test_config_overrides_defaults(self, tmp_path, monkeypatch, caplog):
        """YAML config can disable association marker exclusion."""
        from array_lrr_gwas import association

        cfg_file = tmp_path / "qc.yaml"
        cfg_file.write_text(
            "association_marker_qc:\n"
            "  exclude_monomorphic_lrr: false\n"
        )

        pheno = tmp_path / "pheno.tsv"
        pheno.write_text("sample_id\tphenotype\nS1\t0.1\nS2\t0.2\nS3\t0.3\n")
        out = tmp_path / "results.tsv"
        fake_bcf = tmp_path / "in.bcf"
        fake_bcf.write_text("stub")

        # a2 is monomorphic but should be kept due to config override
        lrr = np.array([[0.1, 0.2, 0.3], [0.5, 0.5, 0.5]], dtype=float)
        samples = ["S1", "S2", "S3"]
        variants = [
            {"chrom": "chr1", "pos": 100, "id": "a1"},
            {"chrom": "chr1", "pos": 200, "id": "a2"},
        ]
        monkeypatch.setattr(
            "array_lrr_gwas.io_vcf.read_lrr",
            lambda _p: (lrr, samples, variants),
        )

        class _FakeResult:
            chrom = ["chr1", "chr1"]

            @staticmethod
            def to_records():
                return [
                    {"chrom": "chr1", "pos": 100, "variant_id": "a1",
                     "beta": 0.0, "se": 1.0, "stat": 0.0, "p_value": 1.0,
                     "n_samples": 3, "method": "ols"},
                    {"chrom": "chr1", "pos": 200, "variant_id": "a2",
                     "beta": 0.0, "se": 1.0, "stat": 0.0, "p_value": 1.0,
                     "n_samples": 3, "method": "ols"},
                ]

        def _check(lrr_sub, phenotype, variants, **_k):
            assert lrr_sub.shape[0] == 2  # monomorphic kept
            return _FakeResult()

        monkeypatch.setattr(association, "run_association", _check)

        rc = main([
            "associate",
            str(fake_bcf),
            "--phenotype", str(pheno),
            "--method", "ols",
            "--config", str(cfg_file),
            "-o", str(out),
        ])
        assert rc == 0
