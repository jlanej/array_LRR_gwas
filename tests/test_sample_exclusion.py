"""Tests for association-stage sample exclusion (classify_samples_for_association).

Covers the full exclusion logic added for GWAS best practice:
  - Core QC (call_rate, lrr_sd)
  - Pre-computed upstream exclusions (pre_pca_excluded, excluded_relatedness,
    excluded_het_outlier)
  - BAF SD (contamination proxy)
  - Sex discordance
  - Extreme inbreeding coefficient
  - Missing/absent optional columns
  - CLI argument parsing for new flags
  - YAML config round-trip for association_qc section
"""

from __future__ import annotations

import pytest

from array_lrr_gwas.sample_sheet import (
    classify_samples_for_association,
    ExclusionResult,
    _parse_bool_field,
)
from array_lrr_gwas.qc_config import defaults, load_config
from array_lrr_gwas.cli import _build_parser


# ---------------------------------------------------------------------------
# Helper: write a sample sheet TSV
# ---------------------------------------------------------------------------

def _write_sheet(tmp_path, rows, cols=None):
    """Write a tab-separated sample sheet from column names and row dicts."""
    if cols is None:
        cols = list(rows[0].keys())
    tsv = tmp_path / "sheet.tsv"
    lines = ["\t".join(cols)]
    for row in rows:
        lines.append("\t".join(str(row.get(c, "")) for c in cols))
    tsv.write_text("\n".join(lines) + "\n")
    return tsv


# ---------------------------------------------------------------------------
# _parse_bool_field
# ---------------------------------------------------------------------------

class TestParseBoolField:
    def test_true_values(self):
        for val in ("true", "True", "TRUE", "1", "yes", "YES"):
            assert _parse_bool_field(val) is True

    def test_false_values(self):
        for val in ("false", "False", "FALSE", "0", "no", "NO"):
            assert _parse_bool_field(val) is False

    def test_none_and_empty(self):
        assert _parse_bool_field(None) is None
        assert _parse_bool_field("") is None
        assert _parse_bool_field("NA") is None
        assert _parse_bool_field("maybe") is None


# ---------------------------------------------------------------------------
# classify_samples_for_association — core QC
# ---------------------------------------------------------------------------

class TestAssociationExclusionCoreQC:
    """Core call-rate + LRR-SD filtering (same as classify_samples_from_sheet)."""

    def test_basic_core_qc(self, tmp_path):
        tsv = _write_sheet(tmp_path, [
            {"Sample_ID": "S1", "call_rate": "0.99", "lrr_sd": "0.10"},  # HQ
            {"Sample_ID": "S2", "call_rate": "0.80", "lrr_sd": "0.10"},  # fail CR
            {"Sample_ID": "S3", "call_rate": "0.99", "lrr_sd": "0.50"},  # fail LRR SD
            {"Sample_ID": "S4", "call_rate": "0.98", "lrr_sd": "0.34"},  # HQ
        ])
        result = classify_samples_for_association(
            tsv, honor_precomputed=False, exclude_baf_sd=False,
            exclude_sex_discordant=False, exclude_extreme_inbreeding=False,
        )
        assert isinstance(result, ExclusionResult)
        assert result.hq_ids == {"S1", "S4"}
        assert result.total == 4
        assert result.counts["low_call_rate"] == 1
        assert result.counts["high_lrr_sd"] == 1
        assert result.counts["total_excluded"] == 2

    def test_boundary_values_at_threshold(self, tmp_path):
        """Samples exactly at threshold boundaries are included."""
        tsv = _write_sheet(tmp_path, [
            {"Sample_ID": "S1", "call_rate": "0.97", "lrr_sd": "0.35"},
        ])
        result = classify_samples_for_association(
            tsv, honor_precomputed=False, exclude_baf_sd=False,
            exclude_sex_discordant=False, exclude_extreme_inbreeding=False,
        )
        assert result.hq_ids == {"S1"}

    def test_non_numeric_values_excluded(self, tmp_path):
        """Samples with non-numeric core QC values are excluded."""
        tsv = _write_sheet(tmp_path, [
            {"Sample_ID": "S1", "call_rate": "0.99", "lrr_sd": "0.10"},
            {"Sample_ID": "S2", "call_rate": "NA", "lrr_sd": "0.10"},
            {"Sample_ID": "S3", "call_rate": "0.99", "lrr_sd": ""},
        ])
        result = classify_samples_for_association(
            tsv, honor_precomputed=False, exclude_baf_sd=False,
            exclude_sex_discordant=False, exclude_extreme_inbreeding=False,
        )
        assert result.hq_ids == {"S1"}

    def test_missing_required_columns_raises(self, tmp_path):
        tsv = _write_sheet(tmp_path, [
            {"Sample_ID": "S1", "PC1": "0.1"},
        ])
        with pytest.raises(ValueError, match="missing required columns"):
            classify_samples_for_association(tsv)

    def test_empty_file_raises(self, tmp_path):
        tsv = tmp_path / "sheet.tsv"
        tsv.write_text("")
        with pytest.raises(ValueError, match="empty"):
            classify_samples_for_association(tsv)


# ---------------------------------------------------------------------------
# Pre-computed upstream exclusions
# ---------------------------------------------------------------------------

class TestPrecomputedExclusions:
    """Honor pre_pca_excluded, excluded_relatedness, excluded_het_outlier."""

    def _base_cols(self):
        return [
            "Sample_ID", "call_rate", "lrr_sd",
            "pre_pca_excluded", "excluded_relatedness", "excluded_het_outlier",
        ]

    def test_pre_pca_excluded(self, tmp_path):
        tsv = _write_sheet(tmp_path, [
            {"Sample_ID": "S1", "call_rate": "0.99", "lrr_sd": "0.10",
             "pre_pca_excluded": "false", "excluded_relatedness": "false",
             "excluded_het_outlier": "false"},
            {"Sample_ID": "S2", "call_rate": "0.99", "lrr_sd": "0.10",
             "pre_pca_excluded": "true", "excluded_relatedness": "false",
             "excluded_het_outlier": "false"},
        ], cols=self._base_cols())
        result = classify_samples_for_association(
            tsv, honor_precomputed=True,
            exclude_baf_sd=False, exclude_sex_discordant=False,
            exclude_extreme_inbreeding=False,
        )
        assert result.hq_ids == {"S1"}
        assert result.counts["pre_pca_excluded"] == 1

    def test_excluded_relatedness(self, tmp_path):
        tsv = _write_sheet(tmp_path, [
            {"Sample_ID": "S1", "call_rate": "0.99", "lrr_sd": "0.10",
             "pre_pca_excluded": "false", "excluded_relatedness": "true",
             "excluded_het_outlier": "false"},
        ], cols=self._base_cols())
        result = classify_samples_for_association(
            tsv, honor_precomputed=True,
            exclude_baf_sd=False, exclude_sex_discordant=False,
            exclude_extreme_inbreeding=False,
        )
        assert result.hq_ids == set()
        assert result.counts["excluded_relatedness"] == 1

    def test_excluded_het_outlier(self, tmp_path):
        tsv = _write_sheet(tmp_path, [
            {"Sample_ID": "S1", "call_rate": "0.99", "lrr_sd": "0.10",
             "pre_pca_excluded": "false", "excluded_relatedness": "false",
             "excluded_het_outlier": "1"},
        ], cols=self._base_cols())
        result = classify_samples_for_association(
            tsv, honor_precomputed=True,
            exclude_baf_sd=False, exclude_sex_discordant=False,
            exclude_extreme_inbreeding=False,
        )
        assert result.hq_ids == set()
        assert result.counts["excluded_het_outlier"] == 1

    def test_honor_precomputed_off(self, tmp_path):
        """When honor_precomputed=False, pre-computed flags are ignored."""
        tsv = _write_sheet(tmp_path, [
            {"Sample_ID": "S1", "call_rate": "0.99", "lrr_sd": "0.10",
             "pre_pca_excluded": "true", "excluded_relatedness": "true",
             "excluded_het_outlier": "true"},
        ], cols=self._base_cols())
        result = classify_samples_for_association(
            tsv, honor_precomputed=False,
            exclude_baf_sd=False, exclude_sex_discordant=False,
            exclude_extreme_inbreeding=False,
        )
        assert result.hq_ids == {"S1"}
        assert result.counts["pre_pca_excluded"] == 0
        assert result.counts["excluded_relatedness"] == 0
        assert result.counts["excluded_het_outlier"] == 0

    def test_missing_precomputed_columns_skipped(self, tmp_path):
        """If columns don't exist, precomputed exclusion is silently skipped."""
        tsv = _write_sheet(tmp_path, [
            {"Sample_ID": "S1", "call_rate": "0.99", "lrr_sd": "0.10"},
        ])
        result = classify_samples_for_association(
            tsv, honor_precomputed=True,
            exclude_baf_sd=False, exclude_sex_discordant=False,
            exclude_extreme_inbreeding=False,
        )
        assert result.hq_ids == {"S1"}


# ---------------------------------------------------------------------------
# BAF SD exclusion
# ---------------------------------------------------------------------------

class TestBafSdExclusion:
    def test_high_baf_sd_excluded(self, tmp_path):
        tsv = _write_sheet(tmp_path, [
            {"Sample_ID": "S1", "call_rate": "0.99", "lrr_sd": "0.10", "baf_sd": "0.10"},
            {"Sample_ID": "S2", "call_rate": "0.99", "lrr_sd": "0.10", "baf_sd": "0.20"},
            {"Sample_ID": "S3", "call_rate": "0.99", "lrr_sd": "0.10", "baf_sd": "0.15"},  # at threshold
        ])
        result = classify_samples_for_association(
            tsv, honor_precomputed=False,
            exclude_baf_sd=True, max_baf_sd=0.15,
            exclude_sex_discordant=False, exclude_extreme_inbreeding=False,
        )
        assert result.hq_ids == {"S1", "S3"}
        assert result.counts["high_baf_sd"] == 1

    def test_baf_sd_off(self, tmp_path):
        tsv = _write_sheet(tmp_path, [
            {"Sample_ID": "S1", "call_rate": "0.99", "lrr_sd": "0.10", "baf_sd": "0.50"},
        ])
        result = classify_samples_for_association(
            tsv, honor_precomputed=False,
            exclude_baf_sd=False,
            exclude_sex_discordant=False, exclude_extreme_inbreeding=False,
        )
        assert result.hq_ids == {"S1"}
        assert result.counts["high_baf_sd"] == 0

    def test_baf_sd_column_missing(self, tmp_path):
        """Missing baf_sd column is silently skipped."""
        tsv = _write_sheet(tmp_path, [
            {"Sample_ID": "S1", "call_rate": "0.99", "lrr_sd": "0.10"},
        ])
        result = classify_samples_for_association(
            tsv, honor_precomputed=False,
            exclude_baf_sd=True, max_baf_sd=0.15,
            exclude_sex_discordant=False, exclude_extreme_inbreeding=False,
        )
        assert result.hq_ids == {"S1"}

    def test_baf_sd_non_numeric_not_excluded(self, tmp_path):
        """Non-numeric baf_sd → treated as not excluded."""
        tsv = _write_sheet(tmp_path, [
            {"Sample_ID": "S1", "call_rate": "0.99", "lrr_sd": "0.10", "baf_sd": "NA"},
        ])
        result = classify_samples_for_association(
            tsv, honor_precomputed=False,
            exclude_baf_sd=True, max_baf_sd=0.15,
            exclude_sex_discordant=False, exclude_extreme_inbreeding=False,
        )
        assert result.hq_ids == {"S1"}


# ---------------------------------------------------------------------------
# Sex discordance exclusion
# ---------------------------------------------------------------------------

class TestSexDiscordantExclusion:
    def test_discordant_excluded(self, tmp_path):
        tsv = _write_sheet(tmp_path, [
            {"Sample_ID": "S1", "call_rate": "0.99", "lrr_sd": "0.10", "sex_status": "OK"},
            {"Sample_ID": "S2", "call_rate": "0.99", "lrr_sd": "0.10", "sex_status": "DISCORDANT"},
            {"Sample_ID": "S3", "call_rate": "0.99", "lrr_sd": "0.10", "sex_status": "discordant"},
        ])
        result = classify_samples_for_association(
            tsv, honor_precomputed=False, exclude_baf_sd=False,
            exclude_sex_discordant=True, exclude_extreme_inbreeding=False,
        )
        assert result.hq_ids == {"S1"}
        assert result.counts["sex_discordant"] == 2

    def test_sex_discordant_off(self, tmp_path):
        tsv = _write_sheet(tmp_path, [
            {"Sample_ID": "S1", "call_rate": "0.99", "lrr_sd": "0.10", "sex_status": "DISCORDANT"},
        ])
        result = classify_samples_for_association(
            tsv, honor_precomputed=False, exclude_baf_sd=False,
            exclude_sex_discordant=False, exclude_extreme_inbreeding=False,
        )
        assert result.hq_ids == {"S1"}
        assert result.counts["sex_discordant"] == 0

    def test_sex_status_column_missing(self, tmp_path):
        tsv = _write_sheet(tmp_path, [
            {"Sample_ID": "S1", "call_rate": "0.99", "lrr_sd": "0.10"},
        ])
        result = classify_samples_for_association(
            tsv, honor_precomputed=False, exclude_baf_sd=False,
            exclude_sex_discordant=True, exclude_extreme_inbreeding=False,
        )
        assert result.hq_ids == {"S1"}


# ---------------------------------------------------------------------------
# Extreme inbreeding F exclusion
# ---------------------------------------------------------------------------

class TestInbreedingExclusion:
    def test_extreme_f_excluded(self, tmp_path):
        tsv = _write_sheet(tmp_path, [
            {"Sample_ID": "S1", "call_rate": "0.99", "lrr_sd": "0.10", "inbreeding_F": "0.05"},
            {"Sample_ID": "S2", "call_rate": "0.99", "lrr_sd": "0.10", "inbreeding_F": "0.20"},
            {"Sample_ID": "S3", "call_rate": "0.99", "lrr_sd": "0.10", "inbreeding_F": "-0.20"},
            {"Sample_ID": "S4", "call_rate": "0.99", "lrr_sd": "0.10", "inbreeding_F": "0.15"},  # at threshold
        ])
        result = classify_samples_for_association(
            tsv, honor_precomputed=False, exclude_baf_sd=False,
            exclude_sex_discordant=False,
            exclude_extreme_inbreeding=True, max_abs_inbreeding_f=0.15,
        )
        assert result.hq_ids == {"S1", "S4"}
        assert result.counts["extreme_inbreeding_f"] == 2

    def test_inbreeding_off(self, tmp_path):
        tsv = _write_sheet(tmp_path, [
            {"Sample_ID": "S1", "call_rate": "0.99", "lrr_sd": "0.10", "inbreeding_F": "0.90"},
        ])
        result = classify_samples_for_association(
            tsv, honor_precomputed=False, exclude_baf_sd=False,
            exclude_sex_discordant=False, exclude_extreme_inbreeding=False,
        )
        assert result.hq_ids == {"S1"}

    def test_inbreeding_column_missing(self, tmp_path):
        tsv = _write_sheet(tmp_path, [
            {"Sample_ID": "S1", "call_rate": "0.99", "lrr_sd": "0.10"},
        ])
        result = classify_samples_for_association(
            tsv, honor_precomputed=False, exclude_baf_sd=False,
            exclude_sex_discordant=False, exclude_extreme_inbreeding=True,
        )
        assert result.hq_ids == {"S1"}

    def test_inbreeding_non_numeric_not_excluded(self, tmp_path):
        tsv = _write_sheet(tmp_path, [
            {"Sample_ID": "S1", "call_rate": "0.99", "lrr_sd": "0.10", "inbreeding_F": "NA"},
        ])
        result = classify_samples_for_association(
            tsv, honor_precomputed=False, exclude_baf_sd=False,
            exclude_sex_discordant=False, exclude_extreme_inbreeding=True,
        )
        assert result.hq_ids == {"S1"}


# ---------------------------------------------------------------------------
# Combined / end-to-end
# ---------------------------------------------------------------------------

class TestCombinedExclusions:
    """Multiple exclusion criteria active simultaneously."""

    _ALL_COLS = [
        "Sample_ID", "call_rate", "lrr_sd",
        "pre_pca_excluded", "excluded_relatedness", "excluded_het_outlier",
        "baf_sd", "sex_status", "inbreeding_F",
    ]

    def test_all_exclusions_active(self, tmp_path):
        tsv = _write_sheet(tmp_path, [
            # HQ: passes everything
            {"Sample_ID": "S1", "call_rate": "0.99", "lrr_sd": "0.10",
             "pre_pca_excluded": "false", "excluded_relatedness": "false",
             "excluded_het_outlier": "false", "baf_sd": "0.05",
             "sex_status": "OK", "inbreeding_F": "0.01"},
            # Excluded: low call rate
            {"Sample_ID": "S2", "call_rate": "0.90", "lrr_sd": "0.10",
             "pre_pca_excluded": "false", "excluded_relatedness": "false",
             "excluded_het_outlier": "false", "baf_sd": "0.05",
             "sex_status": "OK", "inbreeding_F": "0.01"},
            # Excluded: related
            {"Sample_ID": "S3", "call_rate": "0.99", "lrr_sd": "0.10",
             "pre_pca_excluded": "false", "excluded_relatedness": "true",
             "excluded_het_outlier": "false", "baf_sd": "0.05",
             "sex_status": "OK", "inbreeding_F": "0.01"},
            # Excluded: high BAF SD
            {"Sample_ID": "S4", "call_rate": "0.99", "lrr_sd": "0.10",
             "pre_pca_excluded": "false", "excluded_relatedness": "false",
             "excluded_het_outlier": "false", "baf_sd": "0.50",
             "sex_status": "OK", "inbreeding_F": "0.01"},
            # Excluded: sex discordant
            {"Sample_ID": "S5", "call_rate": "0.99", "lrr_sd": "0.10",
             "pre_pca_excluded": "false", "excluded_relatedness": "false",
             "excluded_het_outlier": "false", "baf_sd": "0.05",
             "sex_status": "DISCORDANT", "inbreeding_F": "0.01"},
            # Excluded: extreme inbreeding
            {"Sample_ID": "S6", "call_rate": "0.99", "lrr_sd": "0.10",
             "pre_pca_excluded": "false", "excluded_relatedness": "false",
             "excluded_het_outlier": "false", "baf_sd": "0.05",
             "sex_status": "OK", "inbreeding_F": "0.50"},
            # HQ: another passing sample
            {"Sample_ID": "S7", "call_rate": "0.99", "lrr_sd": "0.10",
             "pre_pca_excluded": "false", "excluded_relatedness": "false",
             "excluded_het_outlier": "false", "baf_sd": "0.05",
             "sex_status": "OK", "inbreeding_F": "0.01"},
        ], cols=self._ALL_COLS)
        result = classify_samples_for_association(tsv)
        assert result.hq_ids == {"S1", "S7"}
        assert result.total == 7
        assert result.counts["low_call_rate"] == 1
        assert result.counts["excluded_relatedness"] == 1
        assert result.counts["high_baf_sd"] == 1
        assert result.counts["sex_discordant"] == 1
        assert result.counts["extreme_inbreeding_f"] == 1
        assert result.counts["total_excluded"] == 5

    def test_sample_with_multiple_exclusion_reasons(self, tmp_path):
        """A sample failing multiple criteria is still counted once in total."""
        tsv = _write_sheet(tmp_path, [
            {"Sample_ID": "S1", "call_rate": "0.80", "lrr_sd": "0.50",
             "pre_pca_excluded": "true", "excluded_relatedness": "true",
             "excluded_het_outlier": "true", "baf_sd": "0.50",
             "sex_status": "DISCORDANT", "inbreeding_F": "0.50"},
        ], cols=self._ALL_COLS)
        result = classify_samples_for_association(tsv)
        assert result.hq_ids == set()
        assert result.counts["total_excluded"] == 1  # only 1 sample total
        assert result.counts["low_call_rate"] == 1
        assert result.counts["high_lrr_sd"] == 1
        assert result.counts["pre_pca_excluded"] == 1
        assert result.counts["excluded_relatedness"] == 1
        assert result.counts["excluded_het_outlier"] == 1
        assert result.counts["high_baf_sd"] == 1
        assert result.counts["sex_discordant"] == 1
        assert result.counts["extreme_inbreeding_f"] == 1

    def test_empty_sheet_returns_empty(self, tmp_path):
        tsv = _write_sheet(tmp_path, [], cols=[
            "Sample_ID", "call_rate", "lrr_sd",
        ])
        result = classify_samples_for_association(
            tsv, honor_precomputed=False, exclude_baf_sd=False,
            exclude_sex_discordant=False, exclude_extreme_inbreeding=False,
        )
        assert result.hq_ids == set()
        assert result.total == 0

    def test_all_exclusions_off_is_core_qc_only(self, tmp_path):
        """With all optional exclusions off, only core QC applies."""
        tsv = _write_sheet(tmp_path, [
            {"Sample_ID": "S1", "call_rate": "0.99", "lrr_sd": "0.10",
             "pre_pca_excluded": "true", "excluded_relatedness": "true",
             "excluded_het_outlier": "true", "baf_sd": "0.50",
             "sex_status": "DISCORDANT", "inbreeding_F": "0.50"},
        ], cols=self._ALL_COLS)
        result = classify_samples_for_association(
            tsv, honor_precomputed=False, exclude_baf_sd=False,
            exclude_sex_discordant=False, exclude_extreme_inbreeding=False,
        )
        assert result.hq_ids == {"S1"}


# ---------------------------------------------------------------------------
# QC config defaults for association_qc
# ---------------------------------------------------------------------------

class TestAssociationQcConfig:
    def test_defaults_has_association_qc(self):
        cfg = defaults()
        assert "association_qc" in cfg
        aqc = cfg["association_qc"]
        assert aqc["honor_precomputed"] is True
        assert aqc["exclude_baf_sd"] is True
        assert aqc["max_baf_sd"] == 0.15
        assert aqc["exclude_sex_discordant"] is True
        assert aqc["exclude_extreme_inbreeding"] is True
        assert aqc["max_abs_inbreeding_f"] == 0.15

    def test_yaml_override(self, tmp_path):
        cfg_file = tmp_path / "qc.yaml"
        cfg_file.write_text(
            "association_qc:\n"
            "  honor_precomputed: false\n"
            "  exclude_baf_sd: false\n"
            "  max_baf_sd: 0.20\n"
            "  exclude_sex_discordant: false\n"
            "  exclude_extreme_inbreeding: false\n"
            "  max_abs_inbreeding_f: 0.20\n"
        )
        cfg = load_config(cfg_file)
        aqc = cfg["association_qc"]
        assert aqc["honor_precomputed"] is False
        assert aqc["exclude_baf_sd"] is False
        assert aqc["max_baf_sd"] == 0.20
        assert aqc["exclude_sex_discordant"] is False
        assert aqc["exclude_extreme_inbreeding"] is False
        assert aqc["max_abs_inbreeding_f"] == 0.20

    def test_partial_yaml_override(self, tmp_path):
        cfg_file = tmp_path / "qc.yaml"
        cfg_file.write_text(
            "association_qc:\n"
            "  max_baf_sd: 0.25\n"
        )
        cfg = load_config(cfg_file)
        aqc = cfg["association_qc"]
        assert aqc["max_baf_sd"] == 0.25
        # Defaults preserved
        assert aqc["honor_precomputed"] is True
        assert aqc["exclude_baf_sd"] is True


# ---------------------------------------------------------------------------
# CLI argument parsing for new exclusion flags
# ---------------------------------------------------------------------------

class TestCliExclusionArgs:
    def test_defaults(self):
        args = _build_parser().parse_args([
            "associate", "in.bcf", "--phenotype", "pheno.tsv", "-o", "out.tsv",
        ])
        assert args.no_honor_precomputed is False
        assert args.no_exclude_baf_sd is False
        assert args.max_baf_sd is None
        assert args.no_exclude_sex_discordant is False
        assert args.no_exclude_extreme_inbreeding is False
        assert args.max_abs_inbreeding_f is None

    def test_disable_flags(self):
        args = _build_parser().parse_args([
            "associate", "in.bcf", "--phenotype", "pheno.tsv", "-o", "out.tsv",
            "--no-honor-precomputed",
            "--no-exclude-baf-sd",
            "--no-exclude-sex-discordant",
            "--no-exclude-extreme-inbreeding",
        ])
        assert args.no_honor_precomputed is True
        assert args.no_exclude_baf_sd is True
        assert args.no_exclude_sex_discordant is True
        assert args.no_exclude_extreme_inbreeding is True

    def test_threshold_overrides(self):
        args = _build_parser().parse_args([
            "associate", "in.bcf", "--phenotype", "pheno.tsv", "-o", "out.tsv",
            "--max-baf-sd", "0.25",
            "--max-abs-inbreeding-f", "0.20",
        ])
        assert args.max_baf_sd == 0.25
        assert args.max_abs_inbreeding_f == 0.20
