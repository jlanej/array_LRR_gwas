"""Tests for the audit trail system and variant QC gating.

Covers:
- AuditLogger recording, summary, and output (TSV, JSON, summary TSV)
- variant_qc_mask audit integration
- in_variant_qc column in association output
- ExclusionResult.excluded_reasons per-sample tracking
- Strict variant QC gating across pipeline stages
"""

from __future__ import annotations

import csv
import json
import logging

import numpy as np
import pytest

from array_lrr_gwas.audit import AuditLogger, AuditRecord
from tests import mock_associate_io


# ---------------------------------------------------------------------------
# AuditLogger unit tests
# ---------------------------------------------------------------------------


class TestAuditLogger:
    """Core AuditLogger functionality."""

    def test_empty_logger(self):
        audit = AuditLogger()
        assert audit.records == []
        assert audit.summary() == []

    def test_record_returns_audit_record(self):
        audit = AuditLogger()
        rec = audit.record(
            stage="test_stage",
            id_type="marker",
            included=["a", "b"],
            excluded={"c": "failed_hwe", "d": "failed_call_rate"},
        )
        assert isinstance(rec, AuditRecord)
        assert rec.stage == "test_stage"
        assert rec.id_type == "marker"
        assert rec.total_input == 4
        assert rec.total_included == 2
        assert rec.total_excluded == 2

    def test_record_groups_by_reason(self):
        audit = AuditLogger()
        rec = audit.record(
            stage="s",
            id_type="marker",
            included=["a"],
            excluded={"b": "r1", "c": "r1", "d": "r2"},
        )
        assert len(rec.excluded_reasons["r1"]) == 2
        assert len(rec.excluded_reasons["r2"]) == 1

    def test_multiple_records(self):
        audit = AuditLogger()
        audit.record(stage="s1", id_type="marker", included=["a"], excluded={})
        audit.record(stage="s2", id_type="sample", included=[], excluded={"x": "r"})
        assert len(audit.records) == 2
        assert audit.records[0].stage == "s1"
        assert audit.records[1].stage == "s2"

    def test_summary(self):
        audit = AuditLogger()
        audit.record(
            stage="s1", id_type="marker",
            included=["a", "b"],
            excluded={"c": "r1", "d": "r1"},
        )
        summary = audit.summary()
        assert len(summary) == 1
        assert summary[0]["stage"] == "s1"
        assert summary[0]["total_included"] == 2
        assert summary[0]["total_excluded"] == 2
        assert summary[0]["excluded_reason_counts"] == {"r1": 2}

    def test_none_excluded_treated_as_empty(self):
        audit = AuditLogger()
        rec = audit.record(
            stage="s", id_type="marker", included=["a"], excluded=None,
        )
        assert rec.total_excluded == 0
        assert rec.excluded_reasons == {}


class TestAuditLoggerOutput:
    """AuditLogger file output methods."""

    def test_write_tsv(self, tmp_path):
        audit = AuditLogger()
        audit.record(
            stage="s1", id_type="marker",
            included=["a", "b"],
            excluded={"c": "failed_hwe", "d": "not_in_variant_qc"},
        )
        path = audit.write_tsv(tmp_path / "audit.tsv")
        assert path.exists()

        with open(path, newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            rows = list(reader)

        # One summary row + two excluded ID rows per stage
        assert len(rows) == 3
        summary_rows = [r for r in rows if r["status"] == "included_summary"]
        excluded_rows = [r for r in rows if r["status"] == "excluded"]
        assert len(summary_rows) == 1
        assert summary_rows[0]["id"] == "n=2"
        assert len(excluded_rows) == 2
        assert {r["id"] for r in excluded_rows} == {"c", "d"}
        reasons = {r["id"]: r["reason"] for r in excluded_rows}
        assert reasons["c"] == "failed_hwe"
        assert reasons["d"] == "not_in_variant_qc"

    def test_write_json(self, tmp_path):
        audit = AuditLogger()
        audit.record(
            stage="s1", id_type="marker",
            included=["a"],
            excluded={"b": "r1"},
        )
        path = audit.write_json(tmp_path / "audit.json")
        assert path.exists()

        with open(path) as fh:
            data = json.load(fh)

        assert len(data["audit_records"]) == 1
        rec = data["audit_records"][0]
        assert rec["stage"] == "s1"
        assert rec["total_included"] == 1
        assert rec["total_excluded"] == 1
        assert rec["excluded_reasons"]["r1"]["count"] == 1
        assert rec["excluded_reasons"]["r1"]["ids"] == ["b"]

    def test_write_summary_tsv(self, tmp_path):
        audit = AuditLogger()
        audit.record(
            stage="s1", id_type="marker",
            included=["a", "b"],
            excluded={"c": "r1"},
        )
        audit.record(
            stage="s2", id_type="sample",
            included=["x"],
            excluded={"y": "r2", "z": "r2"},
        )
        path = audit.write_summary_tsv(tmp_path / "summary.tsv")
        assert path.exists()

        with open(path, newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["stage"] == "s1"
        assert rows[0]["total_included"] == "2"
        assert rows[0]["total_excluded"] == "1"
        assert "included_fraction" in rows[0]
        assert "excluded_fraction" in rows[0]
        assert rows[1]["stage"] == "s2"
        assert rows[1]["total_excluded"] == "2"
        assert "included_fraction" in rows[1]
        assert "excluded_fraction" in rows[1]


class TestAuditLoggerLogging:
    """AuditLogger emits log messages."""

    def test_record_logs_info(self, caplog):
        audit = AuditLogger()
        with caplog.at_level(logging.INFO, logger="array_lrr_gwas.audit"):
            audit.record(
                stage="test_log", id_type="marker",
                included=["a"],
                excluded={"b": "reason_x"},
            )
        assert "Audit [test_log]" in caplog.text
        assert "reason_x" in caplog.text


# ---------------------------------------------------------------------------
# variant_qc_mask audit integration
# ---------------------------------------------------------------------------


class TestVariantQCMaskAudit:
    """Verify variant_qc_mask records audit trail when audit logger provided."""

    def _make_qc_data(self):
        from array_lrr_gwas.variant_qc import VariantQCRecord
        return {
            "v1": VariantQCRecord("v1", True, True, True),
            "v2": VariantQCRecord("v2", False, True, True),   # fail call_rate
            "v3": VariantQCRecord("v3", True, False, True),   # fail hwe
            "v4": VariantQCRecord("v4", True, True, False),   # fail maf (only when required)
        }

    def test_audit_records_included_excluded(self):
        from array_lrr_gwas.variant_qc import variant_qc_mask

        audit = AuditLogger()
        qc_data = self._make_qc_data()
        vids = ["v1", "v2", "v3", "v4"]

        mask = variant_qc_mask(
            vids, qc_data,
            require_call_rate=True, require_hwe=True, require_maf=False,
            audit=audit, audit_stage="test_qc",
        )

        assert len(audit.records) == 1
        rec = audit.records[0]
        assert rec.stage == "test_qc"
        assert rec.id_type == "marker"
        assert rec.total_included == 2  # v1, v4
        assert rec.total_excluded == 2  # v2, v3

    def test_audit_with_maf_required(self):
        from array_lrr_gwas.variant_qc import variant_qc_mask

        audit = AuditLogger()
        qc_data = self._make_qc_data()
        vids = ["v1", "v2", "v3", "v4"]

        mask = variant_qc_mask(
            vids, qc_data,
            require_call_rate=True, require_hwe=True, require_maf=True,
            audit=audit, audit_stage="grm_qc",
        )

        rec = audit.records[0]
        assert rec.total_included == 1  # only v1
        assert rec.total_excluded == 3  # v2, v3, v4

    def test_audit_records_missing_variant_reason(self):
        from array_lrr_gwas.variant_qc import variant_qc_mask

        audit = AuditLogger()
        qc_data = self._make_qc_data()
        vids = ["v1", "v_unknown"]

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mask = variant_qc_mask(
                vids, qc_data,
                require_call_rate=True, require_hwe=True, require_maf=False,
                audit=audit, audit_stage="test_missing",
            )

        rec = audit.records[0]
        assert rec.total_excluded == 1
        assert "not_in_variant_qc" in rec.excluded_reasons

    def test_no_audit_when_none(self):
        """When audit=None, no errors and no audit data produced."""
        from array_lrr_gwas.variant_qc import variant_qc_mask

        qc_data = self._make_qc_data()
        vids = ["v1", "v2"]
        mask = variant_qc_mask(
            vids, qc_data,
            require_call_rate=True, require_hwe=True, require_maf=False,
            audit=None,
        )
        assert mask[0]
        assert not mask[1]

    def test_coverage_warning_when_many_missing(self, caplog):
        """Warning emitted when >10% of variants are absent from QC file."""
        from array_lrr_gwas.variant_qc import variant_qc_mask, VariantQCRecord

        qc_data = {"v1": VariantQCRecord("v1", True, True, True)}
        # 9 out of 10 are missing → 90%
        vids = ["v1"] + [f"missing_{i}" for i in range(9)]

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with caplog.at_level(logging.WARNING, logger="array_lrr_gwas.variant_qc"):
                variant_qc_mask(
                    vids, qc_data,
                    require_call_rate=True, require_hwe=True, require_maf=False,
                )
        assert "absent from the variant QC file" in caplog.text
        assert "build mismatch" in caplog.text.lower() or "mismatch" in caplog.text


# ---------------------------------------------------------------------------
# Subsetting audit integration
# ---------------------------------------------------------------------------


class TestSubsettingAudit:
    """Per-filter audit records from subset_markers()."""

    def test_subset_markers_records_per_filter_audit(self):
        from array_lrr_gwas.subsetting import subset_markers

        audit = AuditLogger()
        rng = np.random.default_rng(42)
        n_markers, n_samples = 20, 5
        lrr = rng.standard_normal((n_markers, n_samples))
        # Force marker 0 to be all NaN (fails call rate + variance)
        lrr[0, :] = np.nan
        vids = [f"v{i}" for i in range(n_markers)]

        mask = subset_markers(
            lrr,
            min_call_rate=0.5,
            min_var=0.001,
            audit=audit,
            variant_ids=vids,
        )

        # At least call_rate and variance stages should be recorded
        stages = [r.stage for r in audit.records]
        assert "correction_marker_call_rate" in stages
        assert "correction_marker_variance" in stages

    def test_subset_markers_no_audit_without_ids(self):
        """audit is ignored when variant_ids is None."""
        from array_lrr_gwas.subsetting import subset_markers

        audit = AuditLogger()
        lrr = np.random.default_rng(42).standard_normal((10, 5))

        mask = subset_markers(lrr, audit=audit, variant_ids=None)
        assert len(audit.records) == 0  # no audit without IDs


# ---------------------------------------------------------------------------
# Correction sample-level audit
# ---------------------------------------------------------------------------


class TestCorrectionSampleAudit:
    """correct_lrr records HQ/LQ sample classification in audit."""

    def test_correct_lrr_records_sample_audit(self):
        from array_lrr_gwas.correction import correct_lrr

        audit = AuditLogger()
        rng = np.random.default_rng(42)
        n_markers, n_samples = 60, 10
        # Low-noise data so most samples pass HQ thresholds
        lrr = rng.standard_normal((n_markers, n_samples)) * 0.1
        # Make sample 0 very noisy (LQ)
        lrr[:, 0] = rng.standard_normal(n_markers) * 5.0
        sample_ids = [f"S{i}" for i in range(n_samples)]
        variant_ids = [f"v{i}" for i in range(n_markers)]

        _, info = correct_lrr(
            lrr, k=1,
            max_lrr_sd=0.35,
            min_sample_call_rate=0.50,
            min_marker_call_rate=0.50,
            min_var=0.0001,
            audit=audit,
            sample_ids=sample_ids,
            variant_ids=variant_ids,
        )

        stages = [r.stage for r in audit.records]
        assert "correction_sample_qc" in stages
        sample_rec = [r for r in audit.records if r.stage == "correction_sample_qc"][0]
        assert sample_rec.id_type == "sample"
        assert sample_rec.total_input == n_samples
        # S0 should be excluded as LQ
        assert sample_rec.total_excluded >= 1


# ---------------------------------------------------------------------------
# ExclusionResult.excluded_reasons
# ---------------------------------------------------------------------------


class TestExclusionResultReasons:
    """ExclusionResult tracks per-sample exclusion reasons."""

    @staticmethod
    def _write_sheet(tmp_path, rows, cols=None):
        import csv as _csv
        if cols is None:
            cols = sorted({k for r in rows for k in r})
        path = tmp_path / "sheet.tsv"
        with open(path, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        return path

    def test_excluded_reasons_populated(self, tmp_path):
        from array_lrr_gwas.sample_sheet import classify_samples_for_association

        tsv = self._write_sheet(tmp_path, [
            {"Sample_ID": "S1", "call_rate": "0.99", "lrr_sd": "0.10"},  # HQ
            {"Sample_ID": "S2", "call_rate": "0.80", "lrr_sd": "0.10"},  # fail call_rate
            {"Sample_ID": "S3", "call_rate": "0.99", "lrr_sd": "0.50"},  # fail lrr_sd
        ], cols=["Sample_ID", "call_rate", "lrr_sd"])

        result = classify_samples_for_association(
            tsv, honor_precomputed=False, exclude_baf_sd=False,
            exclude_sex_discordant=False, exclude_extreme_inbreeding=False,
        )
        assert result.hq_ids == {"S1"}
        assert "S2" in result.excluded_reasons
        assert "low_call_rate" in result.excluded_reasons["S2"]
        assert "S3" in result.excluded_reasons
        assert "high_lrr_sd" in result.excluded_reasons["S3"]
        # S1 should NOT be in excluded_reasons
        assert "S1" not in result.excluded_reasons

    def test_multiple_reasons_per_sample(self, tmp_path):
        from array_lrr_gwas.sample_sheet import classify_samples_for_association

        tsv = self._write_sheet(tmp_path, [
            {"Sample_ID": "S1", "call_rate": "0.80", "lrr_sd": "0.50"},  # fail both
        ], cols=["Sample_ID", "call_rate", "lrr_sd"])

        result = classify_samples_for_association(
            tsv, honor_precomputed=False, exclude_baf_sd=False,
            exclude_sex_discordant=False, exclude_extreme_inbreeding=False,
        )
        reasons = result.excluded_reasons.get("S1", [])
        assert "low_call_rate" in reasons
        assert "high_lrr_sd" in reasons

    def test_hq_samples_have_no_reasons(self, tmp_path):
        from array_lrr_gwas.sample_sheet import classify_samples_for_association

        tsv = self._write_sheet(tmp_path, [
            {"Sample_ID": "S1", "call_rate": "0.99", "lrr_sd": "0.10"},
        ], cols=["Sample_ID", "call_rate", "lrr_sd"])

        result = classify_samples_for_association(
            tsv, honor_precomputed=False, exclude_baf_sd=False,
            exclude_sex_discordant=False, exclude_extreme_inbreeding=False,
        )
        assert result.excluded_reasons == {}


# ---------------------------------------------------------------------------
# in_variant_qc column in association output
# ---------------------------------------------------------------------------


class TestInVariantQcColumn:
    """Verify the in_variant_qc column is added to association output."""

    def test_in_variant_qc_column_present(self, tmp_path, monkeypatch):
        """When --variant-qc is provided, output includes in_variant_qc."""
        from array_lrr_gwas.cli import main

        # Build minimal BCF-like mock
        n_markers, n_samples = 3, 3
        lrr = np.random.default_rng(42).standard_normal((n_markers, n_samples))
        samples = ["S1", "S2", "S3"]
        variants = [
            {"chrom": "chr1", "pos": 100, "id": "a1", "ref": "A", "alts": ("T",)},
            {"chrom": "chr1", "pos": 200, "id": "a2", "ref": "G", "alts": ("C",)},
            {"chrom": "chr1", "pos": 300, "id": "a3", "ref": "T", "alts": ("A",)},
        ]

        class _FakeResult:
            chrom = np.array(["chr1", "chr1", "chr1"])
            variant_id = ["a1", "a2", "a3"]
            p_value = np.array([1.0, 1.0, 1.0])
            stat = np.array([0.0, 0.0, 0.0])
            beta = np.array([0.0, 0.0, 0.0])
            se = np.array([1.0, 1.0, 1.0])

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

        mock_associate_io(monkeypatch, lrr, samples, variants)
        monkeypatch.setattr(
            "array_lrr_gwas.association.run_association_streaming",
            lambda *a, **kw: (_FakeResult(), {
                "n_total": len(variants),
                "n_tested": len(_FakeResult.chrom),
                "n_intensity_only": 0,
                "n_monomorphic": 0,
                "excluded_markers": {},
                "tested_mono_flags": [False] * len(_FakeResult.chrom),
            }),
        )

        # Write phenotype
        pheno = tmp_path / "pheno.tsv"
        pheno.write_text("sample_id\tphenotype\nS1\t1.0\nS2\t2.0\nS3\t3.0\n")

        # Write variant QC (a1 and a2 present, a3 absent)
        qc = tmp_path / "qc.tsv"
        qc.write_text(
            "variant_id\tall_ancestries_call_rate_pass\t"
            "all_ancestries_hwe_pass\tall_ancestries_maf_pass\n"
            "a1\tTrue\tTrue\tTrue\n"
            "a2\tTrue\tFalse\tTrue\n"
        )

        bcf = tmp_path / "fake.bcf"
        bcf.write_bytes(b"")  # not read directly

        out = tmp_path / "results.tsv"
        rc = main([
            "associate", str(bcf),
            "--phenotype", str(pheno),
            "--variant-qc", str(qc),
            "--method", "ols",
            "-o", str(out),
        ])
        assert rc == 0

        with open(out, newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            rows = list(reader)

        assert len(rows) == 3
        # a1: present in QC → True
        assert rows[0]["in_variant_qc"] == "True"
        # a2: present in QC → True (even though it fails HWE)
        assert rows[1]["in_variant_qc"] == "True"
        # a3: absent from QC → False
        assert rows[2]["in_variant_qc"] == "False"

    def test_no_in_variant_qc_without_flag(self, tmp_path, monkeypatch):
        """Without --variant-qc, output does NOT include in_variant_qc."""
        from array_lrr_gwas.cli import main

        n_markers, n_samples = 2, 2
        lrr = np.random.default_rng(42).standard_normal((n_markers, n_samples))
        samples = ["S1", "S2"]
        variants = [
            {"chrom": "chr1", "pos": 100, "id": "a1", "ref": "A", "alts": ("T",)},
            {"chrom": "chr1", "pos": 200, "id": "a2", "ref": "G", "alts": ("C",)},
        ]

        class _FakeResult:
            chrom = np.array(["chr1", "chr1"])
            variant_id = ["a1", "a2"]
            p_value = np.array([1.0, 1.0])
            stat = np.array([0.0, 0.0])
            beta = np.array([0.0, 0.0])
            se = np.array([1.0, 1.0])

            @staticmethod
            def to_records():
                return [
                    {"chrom": "chr1", "pos": 100, "variant_id": "a1",
                     "beta": 0.0, "se": 1.0, "stat": 0.0, "p_value": 1.0,
                     "n_samples": 2, "method": "ols"},
                    {"chrom": "chr1", "pos": 200, "variant_id": "a2",
                     "beta": 0.0, "se": 1.0, "stat": 0.0, "p_value": 1.0,
                     "n_samples": 2, "method": "ols"},
                ]

        mock_associate_io(monkeypatch, lrr, samples, variants)
        monkeypatch.setattr(
            "array_lrr_gwas.association.run_association_streaming",
            lambda *a, **kw: (_FakeResult(), {
                "n_total": len(variants),
                "n_tested": len(_FakeResult.chrom),
                "n_intensity_only": 0,
                "n_monomorphic": 0,
                "excluded_markers": {},
                "tested_mono_flags": [False] * len(_FakeResult.chrom),
            }),
        )

        pheno = tmp_path / "pheno.tsv"
        pheno.write_text("sample_id\tphenotype\nS1\t1.0\nS2\t2.0\n")

        bcf = tmp_path / "fake.bcf"
        bcf.write_bytes(b"")

        out = tmp_path / "results.tsv"
        rc = main([
            "associate", str(bcf),
            "--phenotype", str(pheno),
            "--method", "ols",
            "-o", str(out),
        ])
        assert rc == 0

        with open(out, newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            rows = list(reader)

        assert len(rows) == 2
        # No in_variant_qc column when --variant-qc is not provided
        assert "in_variant_qc" not in rows[0]


# ---------------------------------------------------------------------------
# Audit trail file output via --audit-dir
# ---------------------------------------------------------------------------


class TestAuditDirCli:
    """--audit-dir writes structured audit files."""

    def test_audit_dir_creates_files(self, tmp_path, monkeypatch):
        from array_lrr_gwas.cli import main

        n_markers, n_samples = 3, 3
        lrr = np.random.default_rng(42).standard_normal((n_markers, n_samples))
        samples = ["S1", "S2", "S3"]
        variants = [
            {"chrom": "chr1", "pos": 100, "id": "a1", "ref": "A", "alts": ("T",)},
            {"chrom": "chr1", "pos": 200, "id": "a2", "ref": "G", "alts": ("C",)},
            {"chrom": "chr1", "pos": 300, "id": "a3", "ref": "T", "alts": ("A",)},
        ]

        class _FakeResult:
            chrom = np.array(["chr1", "chr1", "chr1"])
            variant_id = ["a1", "a2", "a3"]
            p_value = np.array([1.0, 1.0, 1.0])
            stat = np.array([0.0, 0.0, 0.0])
            beta = np.array([0.0, 0.0, 0.0])
            se = np.array([1.0, 1.0, 1.0])

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

        mock_associate_io(monkeypatch, lrr, samples, variants)
        monkeypatch.setattr(
            "array_lrr_gwas.association.run_association_streaming",
            lambda *a, **kw: (_FakeResult(), {
                "n_total": len(variants),
                "n_tested": len(_FakeResult.chrom),
                "n_intensity_only": 0,
                "n_monomorphic": 0,
                "excluded_markers": {},
                "tested_mono_flags": [False] * len(_FakeResult.chrom),
            }),
        )

        pheno = tmp_path / "pheno.tsv"
        pheno.write_text("sample_id\tphenotype\nS1\t1.0\nS2\t2.0\nS3\t3.0\n")

        bcf = tmp_path / "fake.bcf"
        bcf.write_bytes(b"")

        audit_dir = tmp_path / "audit"
        out = tmp_path / "results.tsv"

        rc = main([
            "associate", str(bcf),
            "--phenotype", str(pheno),
            "--method", "ols",
            "--audit-dir", str(audit_dir),
            "-o", str(out),
        ])
        assert rc == 0
        assert audit_dir.exists()
        assert (audit_dir / "associate_audit.tsv").exists()
        assert (audit_dir / "associate_audit.json").exists()
        assert (audit_dir / "associate_audit_summary.tsv").exists()

    def _make_fake_result_and_mock(self, monkeypatch, lrr, samples, variants):
        """Helper to set up IO and association mocks."""

        class _FakeResult:
            chrom = np.array(["chr1"] * len(variants))
            variant_id = [v["id"] for v in variants]
            p_value = np.ones(len(variants))
            stat = np.zeros(len(variants))
            beta = np.zeros(len(variants))
            se = np.ones(len(variants))

            @staticmethod
            def to_records():
                return [
                    {"chrom": v["chrom"], "pos": v["pos"], "variant_id": v["id"],
                     "beta": 0.0, "se": 1.0, "stat": 0.0, "p_value": 1.0,
                     "n_samples": len(samples), "method": "ols"}
                    for v in variants
                ]

        mock_associate_io(monkeypatch, lrr, samples, variants)
        monkeypatch.setattr(
            "array_lrr_gwas.association.run_association_streaming",
            lambda *a, **kw: (_FakeResult(), {
                "n_total": len(variants),
                "n_tested": len(variants),
                "n_intensity_only": 0,
                "n_monomorphic": 0,
                "excluded_markers": {},
                "tested_mono_flags": [False] * len(variants),
            }),
        )
        return _FakeResult

    def test_audit_defaults_to_output_dir(self, tmp_path, monkeypatch):
        """Without --audit-dir, audit files appear in the output file's directory."""
        from array_lrr_gwas.cli import main

        n_markers, n_samples = 3, 3
        lrr = np.random.default_rng(0).standard_normal((n_markers, n_samples))
        samples = ["S1", "S2", "S3"]
        variants = [
            {"chrom": "chr1", "pos": 100, "id": "a1", "ref": "A", "alts": ("T",)},
            {"chrom": "chr1", "pos": 200, "id": "a2", "ref": "G", "alts": ("C",)},
            {"chrom": "chr1", "pos": 300, "id": "a3", "ref": "T", "alts": ("A",)},
        ]

        self._make_fake_result_and_mock(monkeypatch, lrr, samples, variants)

        out_subdir = tmp_path / "results"
        out_subdir.mkdir()
        pheno = tmp_path / "pheno.tsv"
        pheno.write_text("sample_id\tphenotype\nS1\t1.0\nS2\t2.0\nS3\t3.0\n")
        bcf = tmp_path / "fake.bcf"
        bcf.write_bytes(b"")
        out = out_subdir / "results.tsv"

        rc = main([
            "associate", str(bcf),
            "--phenotype", str(pheno),
            "--method", "ols",
            "-o", str(out),
        ])
        assert rc == 0
        # Audit files should appear in the output file's directory
        assert (out_subdir / "associate_audit.tsv").exists()
        assert (out_subdir / "associate_audit.json").exists()
        assert (out_subdir / "associate_audit_summary.tsv").exists()

    def test_no_audit_suppresses_audit_files(self, tmp_path, monkeypatch):
        """--no-audit prevents any audit files from being written."""
        from array_lrr_gwas.cli import main

        n_markers, n_samples = 3, 3
        lrr = np.random.default_rng(1).standard_normal((n_markers, n_samples))
        samples = ["S1", "S2", "S3"]
        variants = [
            {"chrom": "chr1", "pos": 100, "id": "a1", "ref": "A", "alts": ("T",)},
            {"chrom": "chr1", "pos": 200, "id": "a2", "ref": "G", "alts": ("C",)},
            {"chrom": "chr1", "pos": 300, "id": "a3", "ref": "T", "alts": ("A",)},
        ]

        self._make_fake_result_and_mock(monkeypatch, lrr, samples, variants)

        pheno = tmp_path / "pheno.tsv"
        pheno.write_text("sample_id\tphenotype\nS1\t1.0\nS2\t2.0\nS3\t3.0\n")
        bcf = tmp_path / "fake.bcf"
        bcf.write_bytes(b"")
        out = tmp_path / "results.tsv"

        rc = main([
            "associate", str(bcf),
            "--phenotype", str(pheno),
            "--method", "ols",
            "--no-audit",
            "-o", str(out),
        ])
        assert rc == 0
        assert not (tmp_path / "associate_audit.tsv").exists()
        assert not (tmp_path / "associate_audit.json").exists()
        assert not (tmp_path / "associate_audit_summary.tsv").exists()


# ---------------------------------------------------------------------------
# CLI --audit-dir argument parsing
# ---------------------------------------------------------------------------


class TestAuditDirArgParsing:
    """--audit-dir is accepted by correct and associate subcommands."""

    def test_correct_audit_dir_parsed(self):
        from array_lrr_gwas.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "correct", "input.bcf", "-o", "out.bcf",
            "--audit-dir", "/tmp/audit",
        ])
        from pathlib import Path
        assert args.audit_dir == Path("/tmp/audit")

    def test_associate_audit_dir_parsed(self):
        from array_lrr_gwas.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "associate", "input.bcf", "--phenotype", "pheno.tsv",
            "-o", "out.tsv", "--audit-dir", "/tmp/audit",
        ])
        from pathlib import Path
        assert args.audit_dir == Path("/tmp/audit")

    def test_audit_dir_defaults_none(self):
        # At parse time audit_dir is None; the runtime default (output dir) is
        # resolved inside _run_associate.
        from array_lrr_gwas.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "associate", "input.bcf", "--phenotype", "pheno.tsv",
            "-o", "out.tsv",
        ])
        assert args.audit_dir is None

    def test_no_audit_flag_parsed(self):
        from array_lrr_gwas.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "associate", "input.bcf", "--phenotype", "pheno.tsv",
            "-o", "out.tsv", "--no-audit",
        ])
        assert args.no_audit is True

    def test_no_audit_flag_defaults_false(self):
        from array_lrr_gwas.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "associate", "input.bcf", "--phenotype", "pheno.tsv",
            "-o", "out.tsv",
        ])
        assert args.no_audit is False


# ---------------------------------------------------------------------------
# Strict gating: variant QC enforced at correction stage
# ---------------------------------------------------------------------------


class TestStrictGatingCorrection:
    """Variant QC mask is applied to RSVD marker selection."""

    def test_variant_qc_mask_passed_to_correct_lrr(
        self, tmp_path, monkeypatch, caplog,
    ):
        """When --variant-qc is supplied, the mask is logged and applied."""
        from array_lrr_gwas.cli import main

        n_markers, n_samples = 5, 4
        lrr = np.random.default_rng(42).standard_normal((n_markers, n_samples))
        samples = [f"S{i}" for i in range(n_samples)]
        variants = [
            {"chrom": "chr1", "pos": i * 100, "id": f"v{i}", "ref": "A", "alts": ("T",)}
            for i in range(n_markers)
        ]

        monkeypatch.setattr(
            "array_lrr_gwas.io_vcf.read_lrr",
            lambda *a, **kw: (lrr.copy(), list(samples), list(variants)),
        )

        _captured_kwargs = {}

        def _fake_correct(lrr_mat, **kw):
            _captured_kwargs.update(kw)
            info = {
                "k": 1,
                "n_hq_samples": n_samples,
                "n_markers_used": n_markers,
                "singular_values": [1.0],
                "sample_scores": np.ones((1, n_samples)),
                "marker_mask": np.ones(n_markers, dtype=bool),
            }
            return lrr_mat, info

        monkeypatch.setattr(
            "array_lrr_gwas.correction.correct_lrr", _fake_correct,
        )
        monkeypatch.setattr(
            "array_lrr_gwas.io_vcf.write_corrected",
            lambda *a, **kw: None,
        )
        monkeypatch.setattr(
            "array_lrr_gwas.genome_build.detect_build",
            lambda *a, **kw: "GRCh38",
        )
        monkeypatch.setattr(
            "array_lrr_gwas.genome_build.get_exclusion_regions",
            lambda *a, **kw: {},
        )

        # Write variant QC file: v0 fails call rate
        qc = tmp_path / "qc.tsv"
        lines = [
            "variant_id\tall_ancestries_call_rate_pass\t"
            "all_ancestries_hwe_pass\tall_ancestries_maf_pass",
        ]
        for i in range(n_markers):
            cr = "False" if i == 0 else "True"
            lines.append(f"v{i}\t{cr}\tTrue\tTrue")
        qc.write_text("\n".join(lines) + "\n")

        bcf = tmp_path / "fake.bcf"
        bcf.write_bytes(b"")

        out = tmp_path / "corrected.bcf"

        with caplog.at_level(logging.INFO, logger="array_lrr_gwas"):
            rc = main([
                "correct", str(bcf), "-o", str(out),
                "--variant-qc", str(qc),
                "--no-complexity-filter",
            ])

        assert rc == 0
        assert "Upstream variant QC (RSVD)" in caplog.text
        # The upstream_qc_mask should have been passed to correct_lrr
        uqm = _captured_kwargs.get("upstream_qc_mask")
        assert uqm is not None
        assert not uqm[0]  # v0 should be masked
        assert all(uqm[1:])  # v1-v4 should pass
