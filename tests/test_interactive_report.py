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
        assert set(m.keys()) == {"SAMPLE", "LRR_SD", "callrate"}
        assert len(m["SAMPLE"]) == 15
        assert len(m["LRR_SD"]) == 15
        assert len(m["callrate"]) == 15

    def test_callrate_with_nans(self, synthetic_lrr, sample_ids):
        from array_lrr_gwas.interactive_report import compute_sample_metrics

        m = compute_sample_metrics(synthetic_lrr, sample_ids)
        # Sample 0 has 2 NaNs out of 100 markers
        assert m["callrate"][0] == pytest.approx(98 / 100, abs=1e-6)
        # Other samples should have callrate = 1.0
        assert m["callrate"][1] == pytest.approx(1.0)


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
        assert lines[0] == "SAMPLE\tLRR_SD\tcallrate"
        assert len(lines) == 16  # header + 15 samples
        # First data line
        parts = lines[1].split("\t")
        assert parts[0] == "S0"
        assert float(parts[1]) > 0
        assert float(parts[2]) <= 1.0


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
