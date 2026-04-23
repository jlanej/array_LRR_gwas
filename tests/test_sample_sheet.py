"""Tests for sample sheet parsing (array_lrr_gwas.sample_sheet)."""

from __future__ import annotations

import numpy as np
import pytest

from array_lrr_gwas.sample_sheet import (
    read_sample_sheet,
    align_samples,
    classify_samples_from_sheet,
    read_all_raw_rows,
    read_sample_sex_map,
)


class TestReadSampleSheet:
    """Tests for ``read_sample_sheet``."""

    def test_basic_parsing(self, tmp_path) -> None:
        tsv = tmp_path / "sheet.tsv"
        tsv.write_text(
            "Sample_ID\tPC1\tPC2\tPC3\n"
            "S1\t0.1\t0.2\t0.3\n"
            "S2\t0.4\t0.5\t0.6\n"
        )
        ids, covs, names = read_sample_sheet(tsv)
        assert ids == ["S1", "S2"]
        assert names == ["PC1", "PC2", "PC3"]
        assert covs.shape == (2, 3)
        np.testing.assert_allclose(covs[0], [0.1, 0.2, 0.3])

    def test_n_pcs_limit(self, tmp_path) -> None:
        tsv = tmp_path / "sheet.tsv"
        cols = "\t".join(f"PC{i}" for i in range(1, 6))
        vals = "\t".join("1.0" for _ in range(5))
        tsv.write_text(f"Sample_ID\t{cols}\nS1\t{vals}\n")
        _, covs, names = read_sample_sheet(tsv, n_pcs=3)
        assert len(names) == 3
        assert covs.shape == (1, 3)
        assert names == ["PC1", "PC2", "PC3"]

    def test_extra_covariates(self, tmp_path) -> None:
        tsv = tmp_path / "sheet.tsv"
        tsv.write_text(
            "Sample_ID\tPC1\tpredicted_sex\n"
            "S1\t0.1\t1\n"
            "S2\t0.2\t0\n"
        )
        _, covs, names = read_sample_sheet(
            tsv, extra_covariates=["predicted_sex"],
        )
        assert names == ["PC1", "predicted_sex"]
        assert covs.shape == (2, 2)

    def test_missing_values(self, tmp_path) -> None:
        tsv = tmp_path / "sheet.tsv"
        tsv.write_text(
            "Sample_ID\tPC1\tPC2\n"
            "S1\t0.1\tNA\n"
            "S2\t\t0.5\n"
        )
        _, covs, _ = read_sample_sheet(tsv)
        assert np.isnan(covs[0, 1])
        assert np.isnan(covs[1, 0])

    def test_empty_raises(self, tmp_path) -> None:
        tsv = tmp_path / "sheet.tsv"
        tsv.write_text("")
        with pytest.raises(ValueError, match="empty"):
            read_sample_sheet(tsv)

    def test_custom_sample_id_col(self, tmp_path) -> None:
        tsv = tmp_path / "sheet.tsv"
        tsv.write_text("IID\tPC1\nS1\t0.1\n")
        ids, _, _ = read_sample_sheet(tsv, sample_id_col="IID")
        assert ids == ["S1"]

    def test_pc_ordering(self, tmp_path) -> None:
        """PCs should be sorted numerically by index, not alphabetically."""
        tsv = tmp_path / "sheet.tsv"
        tsv.write_text("Sample_ID\tPC10\tPC2\tPC1\nS1\t10.0\t2.0\t1.0\n")
        _, covs, names = read_sample_sheet(tsv)
        assert names == ["PC1", "PC2", "PC10"]
        np.testing.assert_allclose(covs[0], [1.0, 2.0, 10.0])


class TestAlignSamples:
    """Tests for ``align_samples``."""

    def test_reorder(self) -> None:
        target = ["B", "A", "C"]
        sheet = ["A", "B", "C"]
        covs = np.array([[1.0], [2.0], [3.0]])
        aligned = align_samples(target, sheet, covs)
        np.testing.assert_allclose(aligned, [[2.0], [1.0], [3.0]])

    def test_missing_filled_with_nan(self) -> None:
        target = ["A", "B", "D"]
        sheet = ["A", "B"]
        covs = np.array([[1.0], [2.0]])
        aligned = align_samples(target, sheet, covs)
        assert aligned.shape == (3, 1)
        assert np.isnan(aligned[2, 0])


class TestClassifySamplesFromSheet:
    """Tests for ``classify_samples_from_sheet``."""

    def test_basic_classification(self, tmp_path) -> None:
        tsv = tmp_path / "sheet.tsv"
        tsv.write_text(
            "Sample_ID\tcall_rate\tlrr_sd\tPC1\n"
            "S1\t0.99\t0.10\t0.1\n"  # HQ
            "S2\t0.80\t0.10\t0.2\n"  # LQ: low call rate
            "S3\t0.99\t0.50\t0.3\n"  # LQ: high lrr_sd
            "S4\t0.98\t0.34\t0.4\n"  # HQ
        )
        hq = classify_samples_from_sheet(tsv)
        assert hq == {"S1", "S4"}

    def test_custom_thresholds(self, tmp_path) -> None:
        tsv = tmp_path / "sheet.tsv"
        tsv.write_text(
            "Sample_ID\tcall_rate\tlrr_sd\n"
            "S1\t0.99\t0.10\n"
            "S2\t0.96\t0.10\n"  # passes default but fails stricter
            "S3\t0.99\t0.30\n"  # passes default but fails stricter
        )
        hq_strict = classify_samples_from_sheet(
            tsv, max_lrr_sd=0.20, min_call_rate=0.98,
        )
        assert hq_strict == {"S1"}

    def test_boundary_values(self, tmp_path) -> None:
        """Samples exactly at threshold are included."""
        tsv = tmp_path / "sheet.tsv"
        tsv.write_text(
            "Sample_ID\tcall_rate\tlrr_sd\n"
            "S1\t0.97\t0.35\n"  # exactly at defaults
        )
        hq = classify_samples_from_sheet(tsv)
        assert hq == {"S1"}

    def test_missing_columns_raises(self, tmp_path) -> None:
        tsv = tmp_path / "sheet.tsv"
        tsv.write_text("Sample_ID\tPC1\nS1\t0.1\n")
        with pytest.raises(ValueError, match="missing required columns"):
            classify_samples_from_sheet(tsv)

    def test_non_numeric_values_skipped(self, tmp_path) -> None:
        tsv = tmp_path / "sheet.tsv"
        tsv.write_text(
            "Sample_ID\tcall_rate\tlrr_sd\n"
            "S1\t0.99\t0.10\n"
            "S2\tNA\t0.10\n"   # non-numeric call_rate
            "S3\t0.99\tNA\n"   # non-numeric lrr_sd
        )
        hq = classify_samples_from_sheet(tsv)
        assert hq == {"S1"}

    def test_empty_sample_sheet(self, tmp_path) -> None:
        tsv = tmp_path / "sheet.tsv"
        tsv.write_text("Sample_ID\tcall_rate\tlrr_sd\n")
        hq = classify_samples_from_sheet(tsv)
        assert hq == set()

    def test_empty_file_raises(self, tmp_path) -> None:
        tsv = tmp_path / "sheet.tsv"
        tsv.write_text("")
        with pytest.raises(ValueError, match="empty"):
            classify_samples_from_sheet(tsv)


class TestReadAllRawRows:
    """Tests for ``read_all_raw_rows``."""

    def test_basic(self, tmp_path) -> None:
        tsv = tmp_path / "sheet.tsv"
        tsv.write_text("sample_id\tcall_rate\tgender\nS1\t0.99\tM\nS2\t0.97\tF\n")
        cols, raw = read_all_raw_rows(tsv)
        assert cols == ["call_rate", "gender"]
        assert set(raw.keys()) == {"S1", "S2"}
        assert raw["S1"]["call_rate"] == "0.99"
        assert raw["S2"]["gender"] == "F"

    def test_case_insensitive_sample_id(self, tmp_path) -> None:
        tsv = tmp_path / "sheet.tsv"
        tsv.write_text("Sample_ID\tcall_rate\nS1\t0.99\n")
        # Default sample_id_col='sample_id' should resolve to 'Sample_ID'
        cols, raw = read_all_raw_rows(tsv)
        assert "Sample_ID" not in cols
        assert "S1" in raw

    def test_missing_rows_excluded(self, tmp_path) -> None:
        tsv = tmp_path / "sheet.tsv"
        tsv.write_text("sample_id\tcall_rate\n\t0.99\nS1\t0.98\n")
        cols, raw = read_all_raw_rows(tsv)
        # Empty sample ID should be skipped
        assert "" not in raw
        assert "S1" in raw

    def test_empty_file_returns_empty(self, tmp_path) -> None:
        tsv = tmp_path / "sheet.tsv"
        tsv.write_text("")
        cols, raw = read_all_raw_rows(tsv)
        assert cols == []
        assert raw == {}

    def test_header_only_returns_empty_raw(self, tmp_path) -> None:
        tsv = tmp_path / "sheet.tsv"
        tsv.write_text("sample_id\tcall_rate\n")
        cols, raw = read_all_raw_rows(tsv)
        assert cols == ["call_rate"]
        assert raw == {}

    def test_illumina_format(self, tmp_path) -> None:
        """Illumina multi-section CSV (comma-delimited with [Header]/[Data]) is parsed correctly."""
        csv_content = (
            "[Header]\n"
            "Investigator Name,Test\n"
            "Date,2024-01-01\n"
            "\n"
            "[Manifests]\n"
            "A,SomeManifest\n"
            "[Data]\n"
            "Sample_ID,SentrixBarcode_A,Gender,CallRate\n"
            "S1,123456,F,0.99\n"
            "S2,789012,M,0.98\n"
        )
        csv_file = tmp_path / "SampleSheet.csv"
        csv_file.write_text(csv_content)
        cols, raw = read_all_raw_rows(csv_file)
        assert cols == ["SentrixBarcode_A", "Gender", "CallRate"]
        assert set(raw.keys()) == {"S1", "S2"}
        assert raw["S1"]["Gender"] == "F"
        assert raw["S2"]["CallRate"] == "0.98"
        # Sample_ID column should not appear in other_columns
        assert "Sample_ID" not in cols

    def test_illumina_format_case_insensitive_data_section(self, tmp_path) -> None:
        """[data] (lowercase) is also recognised."""
        csv_content = (
            "[Header]\n"
            "Project,Test\n"
            "[data]\n"
            "Sample_ID,Gender\n"
            "S1,F\n"
        )
        csv_file = tmp_path / "sheet.csv"
        csv_file.write_text(csv_content)
        cols, raw = read_all_raw_rows(csv_file)
        assert "S1" in raw
        assert raw["S1"]["Gender"] == "F"

    def test_illumina_format_no_data_section_returns_empty(self, tmp_path) -> None:
        """An Illumina-style file without a [Data] section returns empty."""
        csv_content = "[Header]\nProject,Test\n[Manifests]\nA,Manifest\n"
        csv_file = tmp_path / "sheet.csv"
        csv_file.write_text(csv_content)
        cols, raw = read_all_raw_rows(csv_file)
        assert cols == []
        assert raw == {}

    def test_illumina_real_sample_sheet(self) -> None:
        """Parse the real Illumina SampleSheet.csv from tests/data/sampleSheet."""
        import pathlib
        sheet = pathlib.Path(__file__).parent / "data" / "sampleSheet" / "SampleSheet.csv"
        if not sheet.exists():
            pytest.skip("SampleSheet.csv not present in test data")
        cols, raw = read_all_raw_rows(sheet)
        # Known columns in the real sheet
        assert "Gender" in cols
        assert "Sample_Plate" in cols
        assert "CallRate" in cols
        # Sample_ID should not appear as a data column
        assert "Sample_ID" not in cols
        # Should have parsed samples
        assert len(raw) > 0
        # Spot-check a known first-plate sample
        expected_id = "1000G_OMNI2.5M_07-10_01_D05_PT-G23E_5426529111_R04C01"
        if expected_id in raw:
            assert raw[expected_id]["Gender"] == "F"


class TestReadSampleSexMap:
    """Tests for ``read_sample_sex_map`` — the sex-column auto-resolver
    used by sex-chromosome association modes.

    The compiled sample sheet produced by the upstream
    ``illumina_idat_processing`` pipeline does *not* expose a numeric
    ``predicted_sex`` column; sex instead lives in ``computed_gender``,
    ``peddy_sex``, ``peddy_sex_predicted_sex``, or ``f_sex``.  This
    helper normalises those to ``{1: male, 2: female}`` with a
    per-sample priority fallback so downstream sex-chr logic sees the
    real male/female split rather than "0 males, N females".
    """

    def test_predicted_sex_numeric_passthrough(self, tmp_path) -> None:
        tsv = tmp_path / "sheet.tsv"
        tsv.write_text(
            "sample_id\tpredicted_sex\n"
            "S1\t1\nS2\t2\nS3\tNA\n"
        )
        sex_map, primary, counts = read_sample_sex_map(
            tsv, sample_id_col="sample_id"
        )
        assert sex_map == {"S1": 1, "S2": 2}
        assert primary == "predicted_sex"
        assert counts == {"predicted_sex": 2}

    def test_computed_gender_string_codes(self, tmp_path) -> None:
        tsv = tmp_path / "sheet.tsv"
        tsv.write_text(
            "sample_id\tcomputed_gender\n"
            "S1\tM\nS2\tF\nS3\tU\nS4\t\n"
        )
        sex_map, primary, _ = read_sample_sex_map(
            tsv, sample_id_col="sample_id"
        )
        assert sex_map == {"S1": 1, "S2": 2}
        assert primary == "computed_gender"

    def test_peddy_style_words(self, tmp_path) -> None:
        tsv = tmp_path / "sheet.tsv"
        tsv.write_text(
            "sample_id\tpeddy_sex\n"
            "S1\tmale\nS2\tfemale\nS3\tunknown\n"
        )
        sex_map, primary, _ = read_sample_sex_map(
            tsv, sample_id_col="sample_id"
        )
        assert sex_map == {"S1": 1, "S2": 2}
        assert primary == "peddy_sex"

    def test_f_sex_threshold(self, tmp_path) -> None:
        tsv = tmp_path / "sheet.tsv"
        tsv.write_text(
            "sample_id\tf_sex\n"
            "S1\t0.95\nS2\t0.05\nS3\t0.5\nS4\tNA\n"
        )
        sex_map, primary, _ = read_sample_sex_map(
            tsv, sample_id_col="sample_id"
        )
        # 0.95 → male, 0.05 → female, 0.5 is in the ambiguous zone.
        assert sex_map == {"S1": 1, "S2": 2}
        assert primary == "f_sex"

    def test_per_row_fallback_priority(self, tmp_path) -> None:
        """When the highest-priority column is missing for a row, the
        next column should be consulted *for that row only*."""
        tsv = tmp_path / "sheet.tsv"
        tsv.write_text(
            "sample_id\tcomputed_gender\tpeddy_sex\tf_sex\n"
            "S1\tM\tF\t0.1\n"        # computed_gender wins → male
            "S2\tNA\tfemale\t0.9\n"  # fall through to peddy_sex → female
            "S3\tU\tAMBIGUOUS\t0.95\n"  # fall through to f_sex → male
            "S4\tNA\tNA\tNA\n"       # no usable column → omitted
        )
        sex_map, primary, counts = read_sample_sex_map(
            tsv, sample_id_col="sample_id"
        )
        assert sex_map == {"S1": 1, "S2": 2, "S3": 1}
        assert primary == "computed_gender"
        assert counts == {
            "computed_gender": 1,
            "peddy_sex": 1,
            "f_sex": 1,
        }

    def test_case_insensitive_sample_id_column(self, tmp_path) -> None:
        tsv = tmp_path / "sheet.tsv"
        tsv.write_text(
            "SAMPLE_ID\tcomputed_gender\nS1\tM\nS2\tF\n"
        )
        sex_map, _, _ = read_sample_sex_map(
            tsv, sample_id_col="sample_id"
        )
        assert sex_map == {"S1": 1, "S2": 2}

    def test_compiled_sample_sheet_fixture(self) -> None:
        """End-to-end against the bundled compiled sample sheet, which
        has no ``predicted_sex`` column but does have ``computed_gender``,
        ``peddy_sex``, ``peddy_sex_predicted_sex``, and ``f_sex``.
        Regression guard for the bug where the sex-chr pipeline assigned
        0 males / N females because ``predicted_sex`` was missing.
        """
        from pathlib import Path

        sheet = Path(__file__).parent / "data" / "compiled_sample_sheet.tsv"
        sex_map, primary, counts = read_sample_sex_map(
            sheet, sample_id_col="sample_id"
        )
        # The bundled sheet carries both sexes in realistic proportions.
        n_male = sum(1 for v in sex_map.values() if v == 1)
        n_female = sum(1 for v in sex_map.values() if v == 2)
        assert n_male > 500
        assert n_female > 500
        # computed_gender is the highest-priority column actually present.
        assert primary == "computed_gender"
        assert counts.get("computed_gender", 0) > 0
