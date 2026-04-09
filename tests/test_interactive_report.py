"""Tests for the interactive HTML diagnostic report module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_info():
    """Simulated correction info dict with 3 PCs and 15 samples."""
    rng = np.random.default_rng(99)
    n_samples = 15
    k = 3
    return {
        "k": k,
        "singular_values": np.array([8.0, 4.0, 1.5]),
        "sample_scores": rng.standard_normal((k, n_samples)),
        "hq_sample_mask": np.array(
            [True] * 12 + [False] * 3, dtype=bool
        ),
        "n_hq_samples": 12,
        "n_markers_used": 200,
    }


@pytest.fixture()
def synthetic_lrr():
    """Simulated LRR matrix (100 markers × 15 samples)."""
    rng = np.random.default_rng(99)
    lrr = rng.standard_normal((100, 15)) * 0.2
    # Inject a few NaNs to test callrate
    lrr[0, 0] = np.nan
    lrr[1, 0] = np.nan
    return lrr


@pytest.fixture()
def sample_ids():
    return [f"S{i}" for i in range(15)]


# ---------------------------------------------------------------------------
# compute_sample_metrics
# ---------------------------------------------------------------------------

class TestComputeSampleMetrics:
    def test_keys_and_length(self, synthetic_lrr, sample_ids):
        from array_lrr_gwas.interactive_report import compute_sample_metrics

        m = compute_sample_metrics(synthetic_lrr, sample_ids)
        assert set(m.keys()) == {"SAMPLE", "LRR_SD", "callrate", "n_markers_used"}
        assert len(m["SAMPLE"]) == 15
        assert len(m["LRR_SD"]) == 15
        assert len(m["callrate"]) == 15
        assert len(m["n_markers_used"]) == 15

    def test_callrate_with_nans(self, synthetic_lrr, sample_ids):
        from array_lrr_gwas.interactive_report import compute_sample_metrics

        m = compute_sample_metrics(synthetic_lrr, sample_ids)
        # Sample 0 has 2 NaNs out of 100 markers
        assert m["callrate"][0] == pytest.approx(98 / 100, abs=1e-6)
        # Other samples should have callrate = 1.0
        assert m["callrate"][1] == pytest.approx(1.0)

    def test_n_markers_used(self, synthetic_lrr, sample_ids):
        from array_lrr_gwas.interactive_report import compute_sample_metrics

        m = compute_sample_metrics(synthetic_lrr, sample_ids)
        # Sample 0 has 2 NaNs → 98 valid markers
        assert m["n_markers_used"][0] == 98
        # All other samples have no NaNs → 100 valid markers
        assert m["n_markers_used"][1] == 100

    def test_all_nan_sample_lrr_sd_is_none(self, sample_ids):
        """A sample with all-NaN LRR should yield LRR_SD=None (JSON null)."""
        import json
        from array_lrr_gwas.interactive_report import compute_sample_metrics

        rng = np.random.default_rng(42)
        lrr = rng.standard_normal((100, 15)) * 0.2
        # Make sample 3 all-NaN
        lrr[:, 3] = np.nan
        m = compute_sample_metrics(lrr, sample_ids)
        assert m["LRR_SD"][3] is None
        assert m["n_markers_used"][3] == 0
        # Verify the result is JSON-serialisable with allow_nan=False
        encoded = json.dumps({"LRR_SD": m["LRR_SD"]}, allow_nan=False)
        decoded = json.loads(encoded)
        assert decoded["LRR_SD"][3] is None

    def test_inf_values_treated_as_missing(self, sample_ids):
        """Samples with inf LRR values must not produce NaN LRR_SD.

        np.nanstd returns NaN when a column contains inf (inf - mean = NaN),
        even if the majority of values are valid finite floats.  inf values are
        treated the same as NaN: excluded from both LRR_SD and callrate.
        """
        import json
        from array_lrr_gwas.interactive_report import compute_sample_metrics

        rng = np.random.default_rng(7)
        lrr = rng.standard_normal((1000, 15)) * 0.2
        # Sample 5: inject a few inf values (simulates data-quality artefacts
        # or specific BCF float encodings returned by pysam as inf)
        lrr[:5, 5] = np.inf
        # Sample 7: mix of NaN and inf
        lrr[:3, 7] = np.nan
        lrr[3:6, 7] = np.inf

        m = compute_sample_metrics(lrr, sample_ids)

        # LRR_SD must be a valid float (not NaN or None) for both affected samples
        assert m["LRR_SD"][5] is not None
        assert np.isfinite(m["LRR_SD"][5])
        assert m["LRR_SD"][7] is not None
        assert np.isfinite(m["LRR_SD"][7])

        # n_markers_used must exclude both NaN and inf values
        assert m["n_markers_used"][5] == 1000 - 5   # 5 inf excluded
        assert m["n_markers_used"][7] == 1000 - 6   # 3 NaN + 3 inf excluded

        # callrate must also reflect only finite values
        assert m["callrate"][5] == pytest.approx((1000 - 5) / 1000, abs=1e-9)
        assert m["callrate"][7] == pytest.approx((1000 - 6) / 1000, abs=1e-9)

        # Result must be JSON-serialisable with allow_nan=False
        json.dumps({"LRR_SD": m["LRR_SD"]}, allow_nan=False)

    def test_autosomal_only_chromosomes(self, sample_ids):
        """Non-autosomal markers must be excluded from LRR_SD and callrate.

        chrY and chrX markers carry sex-linked signals that inflate LRR_SD.
        When chromosomes are provided only autosomal markers are used.
        """
        from array_lrr_gwas.interactive_report import compute_sample_metrics

        rng = np.random.default_rng(55)
        n_auto = 80
        n_sex = 20  # 20 non-autosomal markers (chrX / chrY)
        n_total = n_auto + n_sex
        lrr = rng.standard_normal((n_total, 15)) * 0.2
        # Inject inf into a sex-chromosome row → would cause NaN LRR_SD if
        # not filtered.
        lrr[n_auto, :] = np.inf  # first chrX row

        chroms = ["chr1"] * n_auto + ["chrX"] * 10 + ["chrY"] * 10
        chroms_arr = np.array(chroms)

        # Without chromosomes: all rows including the inf → may produce NaN
        m_all = compute_sample_metrics(lrr, sample_ids)
        # (isfinite already guards, but callrate denominator includes sex markers)
        assert m_all["n_markers_used"][0] <= n_total

        # With chromosomes: only 80 autosomal rows used
        m_auto = compute_sample_metrics(lrr, sample_ids, chromosomes=chroms_arr)
        assert all(v == n_auto for v in m_auto["n_markers_used"])
        # callrate denominator is now 80 (autosomal only)
        assert all(cr == pytest.approx(1.0) for cr in m_auto["callrate"])
        # All LRR_SD values must be valid floats (no NaN from inf row)
        assert all(v is not None and np.isfinite(v) for v in m_auto["LRR_SD"])


# ---------------------------------------------------------------------------
# write_sample_metrics_tsv
# ---------------------------------------------------------------------------

class TestWriteSampleMetricsTsv:
    def test_round_trip(self, synthetic_lrr, sample_ids, tmp_path):
        from array_lrr_gwas.interactive_report import (
            compute_sample_metrics,
            write_sample_metrics_tsv,
        )

        m = compute_sample_metrics(synthetic_lrr, sample_ids)
        out = write_sample_metrics_tsv(m, tmp_path / "metrics.tsv")
        assert out.exists()
        lines = out.read_text().splitlines()
        assert lines[0] == "SAMPLE\tLRR_SD\tcallrate\tn_markers_used"
        assert len(lines) == 16  # header + 15 samples
        # First data line
        parts = lines[1].split("\t")
        assert parts[0] == "S0"
        assert float(parts[1]) > 0
        assert float(parts[2]) <= 1.0
        assert int(parts[3]) == 98  # 2 NaNs → 98 valid markers

    def test_nan_lrr_sd_written_as_nan(self, sample_ids, tmp_path):
        """Samples with all-NaN LRR should have 'nan' in the LRR_SD column."""
        from array_lrr_gwas.interactive_report import (
            compute_sample_metrics,
            write_sample_metrics_tsv,
        )

        lrr = np.ones((50, 15)) * 0.1
        lrr[:, 5] = np.nan  # sample 5 all-NaN
        m = compute_sample_metrics(lrr, sample_ids)
        out = write_sample_metrics_tsv(m, tmp_path / "metrics_nan.tsv")
        lines = out.read_text().splitlines()
        # Sample index 5 → line 6 (0-based) in data = lines[6]
        parts = lines[6].split("\t")
        assert parts[0] == "S5"
        assert parts[1] == "nan"
        assert int(parts[3]) == 0  # no valid markers


# ---------------------------------------------------------------------------
# Scree data helper
# ---------------------------------------------------------------------------

class TestScreeData:
    def test_scree_structure(self, synthetic_info):
        from array_lrr_gwas.interactive_report import _scree_data

        scree = _scree_data(
            synthetic_info["singular_values"],
            synthetic_info["n_markers_used"],
            synthetic_info["n_hq_samples"],
            synthetic_info["k"],
        )
        assert "eigenvalues" in scree
        assert "prop_var" in scree
        assert "cum_var" in scree
        assert "mp_threshold" in scree
        assert scree["k_mp"] == 3
        assert scree["n_pcs"] == 3
        assert scree["mp_threshold"] > 0
        # Cumulative variance should end at 1.0
        assert scree["cum_var"][-1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# PC scatter data
# ---------------------------------------------------------------------------

class TestPCScatterData:
    def test_scatter_structure(self, synthetic_info, synthetic_lrr, sample_ids):
        from array_lrr_gwas.interactive_report import (
            _pc_scatter_data,
            compute_sample_metrics,
        )

        metrics = compute_sample_metrics(synthetic_lrr, sample_ids)
        scatter = _pc_scatter_data(
            synthetic_info["sample_scores"],
            synthetic_info["singular_values"],
            sample_ids,
            metrics,
            synthetic_info["hq_sample_mask"],
            synthetic_info["k"],
        )
        assert "pcs" in scatter
        assert "PC1" in scatter["pcs"]
        assert len(scatter["pcs"]["PC1"]) == 15
        assert len(scatter["samples"]) == 15
        assert scatter["k_mp"] == 3
        assert len(scatter["hq"]) == 15
        # 12 HQ, 3 LQ
        assert sum(scatter["hq"]) == 12


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------

class TestComputeUMAP:
    def test_umap_output_shape(self, synthetic_info):
        pytest.importorskip("umap")
        from array_lrr_gwas.interactive_report import compute_umap

        u1, u2 = compute_umap(
            synthetic_info["sample_scores"],
            synthetic_info["singular_values"],
            synthetic_info["k"],
        )
        assert len(u1) == 15
        assert len(u2) == 15


# ---------------------------------------------------------------------------
# Full report generation
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_html_output(
        self, synthetic_info, synthetic_lrr, sample_ids, tmp_path
    ):
        from array_lrr_gwas.interactive_report import generate_report

        out = generate_report(
            info=synthetic_info,
            samples=sample_ids,
            lrr=synthetic_lrr,
            output_path=tmp_path / "report.html",
            metrics_tsv_path=tmp_path / "metrics.tsv",
        )
        assert out.exists()
        html = out.read_text()
        assert "<!DOCTYPE html>" in html
        assert "Plotly" in html or "plotly" in html
        assert "scree-plot" in html
        assert "pc-scatter" in html
        assert "umap-plot" in html
        # Metrics TSV should also exist
        assert (tmp_path / "metrics.tsv").exists()

    def test_skip_umap(
        self, synthetic_info, synthetic_lrr, sample_ids, tmp_path
    ):
        from array_lrr_gwas.interactive_report import generate_report

        out = generate_report(
            info=synthetic_info,
            samples=sample_ids,
            lrr=synthetic_lrr,
            output_path=tmp_path / "report_no_umap.html",
            skip_umap=True,
        )
        html = out.read_text()
        assert "umap-plot" in html  # section exists, but data is null
        assert '"umap": null' in html

    def test_no_metrics_tsv(
        self, synthetic_info, synthetic_lrr, sample_ids, tmp_path
    ):
        """When metrics_tsv_path is None, no TSV is written."""
        from array_lrr_gwas.interactive_report import generate_report

        generate_report(
            info=synthetic_info,
            samples=sample_ids,
            lrr=synthetic_lrr,
            output_path=tmp_path / "report.html",
            metrics_tsv_path=None,
        )
        # No metrics file should be created next to the report
        assert not (tmp_path / "metrics.tsv").exists()


# ---------------------------------------------------------------------------
# Sample-sheet column parsing
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"


class TestParseSampleSheetColumns:
    def test_basic_columns(self):
        from array_lrr_gwas.interactive_report import _parse_sample_sheet_columns

        sheet_path = DATA_DIR / "compiled_sample_sheet.tsv"
        # Use first 5 samples from the sheet
        import csv
        with sheet_path.open(newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
        sid_col = next(
            (f for f in fieldnames if f.lower() == "sample_id"), fieldnames[0]
        )
        sample_ids = [r[sid_col] for r in rows[:5]]

        result = _parse_sample_sheet_columns(sheet_path, sample_ids)

        assert "columns" in result
        assert "numeric" in result
        assert "data" in result
        assert len(result["columns"]) == len(result["numeric"])
        # sample ID column must NOT appear in columns list
        assert sid_col not in result["columns"]
        # All data lists must have length == number of samples
        for col, vals in result["data"].items():
            assert len(vals) == len(sample_ids), f"Column {col!r} has wrong length"

    def test_numeric_flag_for_call_rate(self):
        from array_lrr_gwas.interactive_report import _parse_sample_sheet_columns

        sheet_path = DATA_DIR / "compiled_sample_sheet.tsv"
        import csv
        with sheet_path.open(newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            rows = list(reader)
        sample_ids = [r["sample_id"] for r in rows[:10]]

        result = _parse_sample_sheet_columns(sheet_path, sample_ids)

        col_idx = result["columns"].index("call_rate")
        assert result["numeric"][col_idx] is True
        # All call_rate values should be float
        for val in result["data"]["call_rate"]:
            assert val is None or isinstance(val, float)

    def test_sex_status_is_categorical(self):
        from array_lrr_gwas.interactive_report import _parse_sample_sheet_columns

        sheet_path = DATA_DIR / "compiled_sample_sheet.tsv"
        import csv
        with sheet_path.open(newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            rows = list(reader)
        sample_ids = [r["sample_id"] for r in rows[:20]]

        result = _parse_sample_sheet_columns(sheet_path, sample_ids)

        col_idx = result["columns"].index("sex_status")
        assert result["numeric"][col_idx] is False
        # Values should be strings (e.g. "CONCORDANT", "AMBIGUOUS", "DISCORDANT")
        for val in result["data"]["sex_status"]:
            assert isinstance(val, str)

    def test_missing_sample_filled_empty(self):
        from array_lrr_gwas.interactive_report import _parse_sample_sheet_columns

        sheet_path = DATA_DIR / "compiled_sample_sheet.tsv"
        # Include a fake sample ID that doesn't exist in the sheet
        sample_ids = ["FAKE_SAMPLE_XYZ"]
        result = _parse_sample_sheet_columns(sheet_path, sample_ids)

        # All values for the missing sample should be None (numeric) or "" (categorical)
        for col, is_num in zip(result["columns"], result["numeric"]):
            val = result["data"][col][0]
            if is_num:
                assert val is None, f"Expected None for missing numeric {col!r}"
            else:
                assert val == "", f"Expected '' for missing categorical {col!r}"

    def test_empty_sheet_returns_empty(self, tmp_path):
        from array_lrr_gwas.interactive_report import _parse_sample_sheet_columns

        empty = tmp_path / "empty.tsv"
        empty.write_text("")
        result = _parse_sample_sheet_columns(empty, ["S1"])
        assert result == {"columns": [], "numeric": [], "data": {}}

    def test_illumina_sample_sheet_csv_no_error(self):
        """Parsing the real Illumina SampleSheet.csv must not raise AttributeError.

        Rows in that file have fewer fields than headers (trailing commas are
        absent for the last column), so csv.DictReader fills missing values
        with None.  _parse_sample_sheet_columns must handle this gracefully.
        """
        from array_lrr_gwas.interactive_report import _parse_sample_sheet_columns

        sheet = DATA_DIR / "sampleSheet" / "SampleSheet.csv"
        if not sheet.exists():
            pytest.skip("SampleSheet.csv not present in test data")

        # Extract a handful of real sample IDs from the sheet
        import csv
        with sheet.open(newline="", encoding="utf-8") as fh:
            lines = fh.readlines()
        data_start = next(
            (i + 1 for i, ln in enumerate(lines) if ln.strip().lower() == "[data]"),
            None,
        )
        assert data_start is not None, "No [Data] section found in SampleSheet.csv"
        import io
        reader = csv.DictReader(io.StringIO("".join(lines[data_start:])))
        sample_ids = [row["Sample_ID"] for row in reader if row.get("Sample_ID")]

        # Must not raise
        result = _parse_sample_sheet_columns(sheet, sample_ids[:20])

        assert "columns" in result
        assert "numeric" in result
        assert "data" in result
        # CallRate column must be present and recognised as numeric
        assert "CallRate" in result["columns"]
        cr_idx = result["columns"].index("CallRate")
        assert result["numeric"][cr_idx] is True, "CallRate should be numeric"
        # All data lists must have correct length
        for col, vals in result["data"].items():
            assert len(vals) == min(len(sample_ids), 20), (
                f"Column {col!r} has wrong length"
            )


class TestGenerateReportWithSampleSheet:
    def _make_info(self, n_samples: int = 10, k: int = 3) -> dict:
        rng = np.random.default_rng(42)
        return {
            "k": k,
            "singular_values": np.array([8.0, 4.0, 1.5]),
            "sample_scores": rng.standard_normal((k, n_samples)),
            "hq_sample_mask": np.ones(n_samples, dtype=bool),
            "n_hq_samples": n_samples,
            "n_markers_used": 200,
        }

    def test_report_embeds_sheet_data(self, tmp_path):
        """generate_report embeds sample sheet columns in the HTML."""
        from array_lrr_gwas.interactive_report import generate_report

        sheet_path = DATA_DIR / "compiled_sample_sheet.tsv"
        import csv
        with sheet_path.open(newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            rows = list(reader)
        sample_ids = [r["sample_id"] for r in rows[:10]]

        rng = np.random.default_rng(7)
        lrr = rng.standard_normal((100, 10)) * 0.2
        info = self._make_info(n_samples=10)

        out = generate_report(
            info=info,
            samples=sample_ids,
            lrr=lrr,
            output_path=tmp_path / "report_sheet.html",
            sample_sheet_path=sheet_path,
            skip_umap=True,
        )
        assert out.exists()
        html = out.read_text()
        # Sheet data should be embedded in JSON
        assert '"sheet"' in html
        assert '"call_rate"' in html
        assert '"sex_status"' in html
        # JS should add sheet-prefixed colour options
        assert "sheet:" in html

    def test_report_without_sheet_has_null_sheet(self, tmp_path):
        """Without sample_sheet_path, the sheet key is null."""
        from array_lrr_gwas.interactive_report import generate_report

        rng = np.random.default_rng(7)
        lrr = rng.standard_normal((100, 5)) * 0.2
        info = self._make_info(n_samples=5)

        out = generate_report(
            info=info,
            samples=[f"S{i}" for i in range(5)],
            lrr=lrr,
            output_path=tmp_path / "report_no_sheet.html",
            skip_umap=True,
        )
        html = out.read_text()
        assert '"sheet": null' in html

    def test_invalid_sheet_path_warns_not_crashes(self, tmp_path):
        """A missing sample sheet path produces a warning but no exception."""
        from array_lrr_gwas.interactive_report import generate_report

        rng = np.random.default_rng(7)
        lrr = rng.standard_normal((100, 5)) * 0.2
        info = self._make_info(n_samples=5)

        # Should succeed even with a non-existent path (warning logged instead)
        out = generate_report(
            info=info,
            samples=[f"S{i}" for i in range(5)],
            lrr=lrr,
            output_path=tmp_path / "report_bad_sheet.html",
            sample_sheet_path=tmp_path / "nonexistent.tsv",
            skip_umap=True,
        )
        assert out.exists()
        html = out.read_text()
        assert '"sheet": null' in html


class TestMergeSheetData:
    """Tests for ``_merge_sheet_data``."""

    def _make_sheet(self, cols, numeric, data):
        return {"columns": cols, "numeric": numeric, "data": data}

    def test_no_collision(self):
        from array_lrr_gwas.interactive_report import _merge_sheet_data

        primary = self._make_sheet(["a"], [True], {"a": [1.0]})
        secondary = self._make_sheet(["b"], [False], {"b": ["X"]})
        merged = _merge_sheet_data(primary, secondary)
        assert merged["columns"] == ["a", "b"]
        assert merged["numeric"] == [True, False]
        assert merged["data"] == {"a": [1.0], "b": ["X"]}

    def test_collision_suffixes_secondary(self):
        from array_lrr_gwas.interactive_report import _merge_sheet_data

        primary = self._make_sheet(["Gender"], [False], {"Gender": ["F"]})
        secondary = self._make_sheet(["Gender", "Plate"], [False, False], {"Gender": ["M"], "Plate": ["P1"]})
        merged = _merge_sheet_data(primary, secondary)
        assert "Gender" in merged["columns"]
        assert "Gender_illumina" in merged["columns"]
        assert "Plate" in merged["columns"]
        assert merged["data"]["Gender"] == ["F"]
        assert merged["data"]["Gender_illumina"] == ["M"]

    def test_primary_empty(self):
        from array_lrr_gwas.interactive_report import _merge_sheet_data

        primary = self._make_sheet([], [], {})
        secondary = self._make_sheet(["x"], [True], {"x": [1.0]})
        merged = _merge_sheet_data(primary, secondary)
        assert merged["columns"] == ["x"]
        assert merged["data"]["x"] == [1.0]


class TestGenerateReportWithIlluminaSheet:
    """generate_report with --illumina-sample-sheet (and combined)."""

    def _make_info(self, n_samples: int = 5, k: int = 2) -> dict:
        rng = np.random.default_rng(11)
        return {
            "k": k,
            "singular_values": np.array([5.0, 2.0]),
            "sample_scores": rng.standard_normal((k, n_samples)),
            "hq_sample_mask": np.ones(n_samples, dtype=bool),
            "n_hq_samples": n_samples,
            "n_markers_used": 100,
        }

    def _make_illumina_sheet(self, tmp_path, samples):
        content = (
            "[Header]\nProject,Test\n[Data]\n"
            "Sample_ID,Gender,Sample_Plate\n"
        )
        for i, sid in enumerate(samples):
            content += f"{sid},{'MF'[i % 2]},Plate{i // 2 + 1}\n"
        p = tmp_path / "SampleSheet.csv"
        p.write_text(content)
        return p

    def test_illumina_only_embeds_columns(self, tmp_path):
        from array_lrr_gwas.interactive_report import generate_report

        samples = [f"S{i}" for i in range(5)]
        illum = self._make_illumina_sheet(tmp_path, samples)
        rng = np.random.default_rng(11)
        lrr = rng.standard_normal((50, 5)) * 0.2
        out = generate_report(
            info=self._make_info(),
            samples=samples,
            lrr=lrr,
            output_path=tmp_path / "rep_illum.html",
            illumina_sample_sheet_path=illum,
            skip_umap=True,
        )
        html = out.read_text()
        assert '"sheet"' in html
        assert '"Gender"' in html
        assert '"Sample_Plate"' in html

    def test_both_sheets_merged(self, tmp_path):
        from array_lrr_gwas.interactive_report import generate_report

        samples = [f"S{i}" for i in range(5)]
        illum = self._make_illumina_sheet(tmp_path, samples)

        # Minimal compiled TSV
        tsv = tmp_path / "compiled.tsv"
        rows = "sample_id\tcall_rate\n" + "".join(f"{s}\t0.{99 - i}\n" for i, s in enumerate(samples))
        tsv.write_text(rows)

        rng = np.random.default_rng(11)
        lrr = rng.standard_normal((50, 5)) * 0.2
        out = generate_report(
            info=self._make_info(),
            samples=samples,
            lrr=lrr,
            output_path=tmp_path / "rep_both.html",
            sample_sheet_path=tsv,
            illumina_sample_sheet_path=illum,
            skip_umap=True,
        )
        html = out.read_text()
        # Both sources should be represented
        assert '"call_rate"' in html
        assert '"Gender"' in html
        assert '"Sample_Plate"' in html

    def test_both_sheets_collision_resolved(self, tmp_path):
        from array_lrr_gwas.interactive_report import generate_report

        samples = ["S0", "S1"]
        illum = self._make_illumina_sheet(tmp_path, samples)

        # Compiled sheet that also has a 'Gender' column → collision
        tsv = tmp_path / "compiled.tsv"
        tsv.write_text("sample_id\tGender\nS0\tFemale\nS1\tMale\n")

        rng = np.random.default_rng(11)
        lrr = rng.standard_normal((50, 2)) * 0.2
        out = generate_report(
            info=self._make_info(n_samples=2),
            samples=samples,
            lrr=lrr,
            output_path=tmp_path / "rep_collision.html",
            sample_sheet_path=tsv,
            illumina_sample_sheet_path=illum,
            skip_umap=True,
        )
        html = out.read_text()
        # Primary 'Gender' and suffixed 'Gender_illumina' both present
        assert '"Gender"' in html
        assert '"Gender_illumina"' in html

    def test_invalid_illumina_path_warns_not_crashes(self, tmp_path):
        from array_lrr_gwas.interactive_report import generate_report

        rng = np.random.default_rng(11)
        lrr = rng.standard_normal((50, 5)) * 0.2
        out = generate_report(
            info=self._make_info(),
            samples=[f"S{i}" for i in range(5)],
            lrr=lrr,
            output_path=tmp_path / "rep_bad_illum.html",
            illumina_sample_sheet_path=tmp_path / "nonexistent.csv",
            skip_umap=True,
        )
        assert out.exists()
        html = out.read_text()
        assert '"sheet": null' in html


# ---------------------------------------------------------------------------
# JSON serialisation edge cases
# ---------------------------------------------------------------------------

class TestJsonDefault:
    def test_numpy_types(self):
        from array_lrr_gwas.interactive_report import _json_default

        assert _json_default(np.int64(5)) == 5
        assert _json_default(np.float64(3.14)) == pytest.approx(3.14)
        assert _json_default(np.bool_(True)) is True
        assert _json_default(np.array([1, 2])) == [1, 2]

    def test_unsupported_type(self):
        from array_lrr_gwas.interactive_report import _json_default

        with pytest.raises(TypeError):
            _json_default(object())


# ---------------------------------------------------------------------------
# Regression: NaN LRR_SD from all-NaN samples must not crash JSON serialisation
# ---------------------------------------------------------------------------

class TestNanLrrSdRegression:
    """Regression tests for samples with all-NaN LRR values (no valid markers).

    These reproduce the ValueError raised by json.dumps(allow_nan=False) when
    np.nanstd returns nan for an all-NaN column.
    """

    def _make_info(self, n_samples: int = 15, k: int = 3) -> dict:
        rng = np.random.default_rng(7)
        return {
            "k": k,
            "singular_values": np.array([8.0, 4.0, 1.5]),
            "sample_scores": rng.standard_normal((k, n_samples)),
            "hq_sample_mask": np.array([True] * (n_samples - 3) + [False] * 3, dtype=bool),
            "n_hq_samples": n_samples - 3,
            "n_markers_used": 200,
        }

    def test_generate_report_with_nan_lrr_sd(self, tmp_path):
        """generate_report must succeed when some samples have all-NaN LRR."""
        from array_lrr_gwas.interactive_report import generate_report

        rng = np.random.default_rng(42)
        n_samples = 15
        lrr = rng.standard_normal((200, n_samples)) * 0.2
        # Make 5 samples entirely NaN (simulates real-world missing data)
        for col in [1, 3, 7, 9, 13]:
            lrr[:, col] = np.nan
        sample_ids = [f"S{i}" for i in range(n_samples)]
        info = self._make_info(n_samples)

        out = generate_report(
            info=info,
            samples=sample_ids,
            lrr=lrr,
            output_path=tmp_path / "report_nan.html",
            metrics_tsv_path=tmp_path / "metrics_nan.tsv",
            skip_umap=True,
        )
        assert out.exists()
        html = out.read_text()
        # null (not NaN) must appear in the JSON blob for all-NaN samples
        assert '"LRR_SD"' in html
        # Metrics TSV must contain 'nan' for the affected samples
        tsv_lines = (tmp_path / "metrics_nan.tsv").read_text().splitlines()
        assert tsv_lines[0].startswith("SAMPLE\tLRR_SD\tcallrate\tn_markers_used")
        nan_samples = {
            parts[0]
            for line in tsv_lines[1:]
            for parts in [line.split("\t")]
            if parts[1] == "nan"
        }
        assert nan_samples == {"S1", "S3", "S7", "S9", "S13"}

    def test_debug_sample_report_data(self, tmp_path):
        """The debug data from tests/data/debugSampleReport must load and produce a valid report."""
        import csv
        from array_lrr_gwas.interactive_report import generate_report

        data_dir = Path(__file__).parent / "data" / "debugSampleReport"
        metrics_path = data_dir / "stage2_reclustered.corrected.bcf.svd.sample_metrics.tsv"
        pcs_path = data_dir / "stage2_reclustered.corrected.bcf.svd.sample_pcs.tsv"
        sv_path = data_dir / "stage2_reclustered.corrected.bcf.svd.singular_values.tsv"

        # Load singular values
        sv_lines = sv_path.read_text().splitlines()
        singular_values = np.array([float(line.split("\t")[1]) for line in sv_lines[1:]])

        # Load sample PCs
        with pcs_path.open() as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            rows = list(reader)
        samples = [r["SAMPLE"] for r in rows]
        pc_cols = [c for c in rows[0] if c.startswith("PC")]
        n_pcs = len(pc_cols)
        n_samples = len(samples)
        # sample_scores shape: (k, n_samples)
        sample_scores = np.array(
            [[float(r[pc]) for r in rows] for pc in pc_cols]
        )

        k = n_pcs
        # Build a minimal info dict
        info = {
            "k": k,
            "singular_values": singular_values[:k],
            "sample_scores": sample_scores,
            "hq_sample_mask": np.ones(n_samples, dtype=bool),
            "n_hq_samples": n_samples,
            "n_markers_used": 100_000,
        }

        # Build a fake LRR matrix matching the metrics (we just need NaN pattern)
        # Use a constant matrix; the metrics TSV already has the NaN pattern
        lrr = np.ones((100, n_samples)) * 0.15

        out = generate_report(
            info=info,
            samples=samples,
            lrr=lrr,
            output_path=tmp_path / "debug_report.html",
            skip_umap=True,
        )
        assert out.exists()
        html = out.read_text()
        assert "<!DOCTYPE html>" in html


# ---------------------------------------------------------------------------
# Post-correction LRR metrics
# ---------------------------------------------------------------------------


class TestComputeSampleMetricsWithUpstreamQc:
    """compute_sample_metrics with upstream_qc_mask restricts markers."""

    def test_qc_mask_restricts_markers(self, sample_ids):
        from array_lrr_gwas.interactive_report import compute_sample_metrics

        rng = np.random.default_rng(12)
        lrr = rng.standard_normal((100, 15)) * 0.2
        # Upstream QC mask: first 60 markers pass, rest fail
        qc_mask = np.array([True] * 60 + [False] * 40)
        m_all = compute_sample_metrics(lrr, sample_ids)
        m_qc = compute_sample_metrics(lrr, sample_ids, upstream_qc_mask=qc_mask)
        # With QC mask, denominator is 60 (not 100)
        assert all(v == 60 for v in m_qc["n_markers_used"])
        # Without QC mask, denominator is 100
        assert all(v == 100 for v in m_all["n_markers_used"])

    def test_qc_mask_with_chromosomes(self, sample_ids):
        """QC mask AND autosomal filter are combined."""
        from array_lrr_gwas.interactive_report import compute_sample_metrics

        rng = np.random.default_rng(13)
        lrr = rng.standard_normal((100, 15)) * 0.2
        chroms = np.array(["chr1"] * 80 + ["chrX"] * 20)
        # QC mask: first 50 pass, rest fail
        qc_mask = np.array([True] * 50 + [False] * 50)
        m = compute_sample_metrics(
            lrr, sample_ids, chromosomes=chroms, upstream_qc_mask=qc_mask,
        )
        # Autosomal markers: 80. QC passing: first 50. Intersection: 50.
        assert all(v == 50 for v in m["n_markers_used"])


class TestWriteSampleMetricsTsvWithPost:
    """write_sample_metrics_tsv includes post-correction columns."""

    def test_post_columns_present(self, synthetic_lrr, sample_ids, tmp_path):
        from array_lrr_gwas.interactive_report import (
            compute_sample_metrics,
            write_sample_metrics_tsv,
        )

        pre = compute_sample_metrics(synthetic_lrr, sample_ids)
        # Simulate corrected LRR (slightly reduced noise)
        corrected_lrr = synthetic_lrr * 0.8
        post = compute_sample_metrics(corrected_lrr, sample_ids)
        out = write_sample_metrics_tsv(pre, tmp_path / "m.tsv", post_metrics=post)
        lines = out.read_text().splitlines()
        assert "LRR_SD_post" in lines[0]
        assert "callrate_post" in lines[0]
        # Each data row should have 6 columns (including post)
        for line in lines[1:]:
            parts = line.split("\t")
            assert len(parts) == 6

    def test_no_post_metrics_backward_compat(self, synthetic_lrr, sample_ids, tmp_path):
        """Without post_metrics, output is unchanged."""
        from array_lrr_gwas.interactive_report import (
            compute_sample_metrics,
            write_sample_metrics_tsv,
        )

        pre = compute_sample_metrics(synthetic_lrr, sample_ids)
        out = write_sample_metrics_tsv(pre, tmp_path / "m.tsv")
        lines = out.read_text().splitlines()
        assert "LRR_SD_post" not in lines[0]
        assert lines[0] == "SAMPLE\tLRR_SD\tcallrate\tn_markers_used"


class TestLrrComparisonData:
    """_lrr_comparison_data returns correct structure."""

    def test_keys_and_length(self, sample_ids):
        from array_lrr_gwas.interactive_report import _lrr_comparison_data

        pre = {
            "SAMPLE": sample_ids,
            "LRR_SD": [0.1 * i for i in range(15)],
            "callrate": [0.99] * 15,
        }
        post = {
            "SAMPLE": sample_ids,
            "LRR_SD": [0.05 * i for i in range(15)],
            "callrate": [0.99] * 15,
        }
        hq_mask = np.array([True] * 12 + [False] * 3, dtype=bool)
        comp = _lrr_comparison_data(pre, post, hq_mask)
        assert set(comp.keys()) == {
            "samples", "LRR_SD_pre", "LRR_SD_post",
            "callrate_pre", "callrate_post", "hq",
        }
        assert len(comp["samples"]) == 15
        assert len(comp["hq"]) == 15
        assert comp["LRR_SD_pre"] == pre["LRR_SD"]
        assert comp["LRR_SD_post"] == post["LRR_SD"]


class TestGenerateReportWithCorrectedLrr:
    """generate_report with corrected_lrr produces comparison plots."""

    def test_comparison_in_html(
        self, synthetic_info, synthetic_lrr, sample_ids, tmp_path
    ):
        from array_lrr_gwas.interactive_report import generate_report

        corrected = synthetic_lrr * 0.7  # simulate reduced noise
        out = generate_report(
            info=synthetic_info,
            samples=sample_ids,
            lrr=synthetic_lrr,
            corrected_lrr=corrected,
            output_path=tmp_path / "report_post.html",
            metrics_tsv_path=tmp_path / "metrics_post.tsv",
            skip_umap=True,
        )
        html = out.read_text()
        assert "lrr-scatter" in html
        assert "bland-altman" in html
        assert '"lrr_comparison"' in html
        assert '"lrr_comparison": null' not in html

    def test_metrics_tsv_has_post_columns(
        self, synthetic_info, synthetic_lrr, sample_ids, tmp_path
    ):
        from array_lrr_gwas.interactive_report import generate_report

        corrected = synthetic_lrr * 0.7
        generate_report(
            info=synthetic_info,
            samples=sample_ids,
            lrr=synthetic_lrr,
            corrected_lrr=corrected,
            output_path=tmp_path / "report.html",
            metrics_tsv_path=tmp_path / "metrics.tsv",
            skip_umap=True,
        )
        tsv = (tmp_path / "metrics.tsv").read_text()
        header = tsv.splitlines()[0]
        assert "LRR_SD_post" in header
        assert "callrate_post" in header

    def test_no_corrected_lrr_no_comparison(
        self, synthetic_info, synthetic_lrr, sample_ids, tmp_path
    ):
        """Without corrected_lrr, comparison data is null."""
        from array_lrr_gwas.interactive_report import generate_report

        out = generate_report(
            info=synthetic_info,
            samples=sample_ids,
            lrr=synthetic_lrr,
            output_path=tmp_path / "report_no_post.html",
            skip_umap=True,
        )
        html = out.read_text()
        assert '"lrr_comparison": null' in html

    def test_post_lrr_sd_lower_after_correction(
        self, synthetic_info, sample_ids, tmp_path
    ):
        """Post-correction LRR_SD should be lower when noise is reduced."""
        from array_lrr_gwas.interactive_report import generate_report

        rng = np.random.default_rng(42)
        lrr = rng.standard_normal((100, 15)) * 0.3
        corrected = lrr * 0.5  # reduce noise
        generate_report(
            info=synthetic_info,
            samples=sample_ids,
            lrr=lrr,
            corrected_lrr=corrected,
            output_path=tmp_path / "report.html",
            metrics_tsv_path=tmp_path / "metrics.tsv",
            skip_umap=True,
        )
        lines = (tmp_path / "metrics.tsv").read_text().splitlines()
        for line in lines[1:]:
            parts = line.split("\t")
            pre_sd = float(parts[1])
            post_sd = float(parts[4])
            if pre_sd > 0:
                assert post_sd < pre_sd
