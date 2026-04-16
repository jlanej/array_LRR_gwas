"""Tests for the gwas_report module."""

from __future__ import annotations

import csv
import gzip
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from array_lrr_gwas import gwas_report
from array_lrr_gwas.gwas_report import (
    annotate_hits_with_genes,
    build_manhattan_figure,
    build_qq_figure,
    build_regional_figure,
    genes_in_region,
    generate_gwas_report,
    lambda_gc,
    load_gene_table,
    read_association_records,
    summarize_mode,
    top_hits,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_fake_refgene(cache: Path, build: str = "GRCh38") -> Path:
    """Write a minimal UCSC refGene-format file into cache."""
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{build}_refGene.txt.gz"
    rows = [
        # bin name chrom strand txStart txEnd cdsStart cdsEnd exonCount exonStarts exonEnds score name2 ...
        ["0", "NM_001", "chr1", "+", "99", "200", "105", "195", "1",
         "100,", "200,", "0", "GENEA", "cmpl", "cmpl", "0,"],
        ["0", "NM_002", "chr1", "-", "499", "800", "510", "790", "1",
         "500,", "800,", "0", "GENEB", "cmpl", "cmpl", "0,"],
        ["0", "NM_003", "chr2", "+", "0", "1000", "10", "990", "1",
         "0,", "1000,", "0", "GENEC", "cmpl", "cmpl", "0,"],
        # A duplicate transcript of GENEA with an overlapping span (tests span merge).
        ["0", "NM_001b", "chr1", "+", "120", "250", "125", "245", "1",
         "121,", "250,", "0", "GENEA", "cmpl", "cmpl", "0,"],
    ]
    with gzip.open(path, "wt") as fh:
        for r in rows:
            fh.write("\t".join(r) + "\n")
    return path


def _write_results_tsv(path: Path, records: list[dict]) -> None:
    cols = ["chrom", "pos", "variant_id", "beta", "se", "stat",
            "p_value", "n_samples", "method"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in records:
            w.writerow({c: r.get(c, "") for c in cols})


def _synthetic_records(
    n_per_chrom: int = 50,
    chroms=("1", "2"),
    planted=((("1", 150), 1e-12),),
):
    """Build synthetic per-marker association records."""
    rng = np.random.default_rng(0)
    out: list[dict] = []
    planted_map = dict(planted)
    for c in chroms:
        for i in range(n_per_chrom):
            pos = 100 + i * 10
            pv = planted_map.get((c, pos), float(rng.uniform(0.1, 1.0)))
            out.append({
                "chrom": c,
                "pos": pos,
                "variant_id": f"{c}:{pos}",
                "beta": float(rng.normal(0, 0.01)),
                "se": 0.1,
                "stat": 0.1,
                "p_value": pv,
                "n_samples": 100,
                "method": "ols",
            })
    return out


# ---------------------------------------------------------------------------
# lambda_gc
# ---------------------------------------------------------------------------


class TestLambdaGC:
    def test_null_uniform_close_to_one(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(1e-8, 1.0, 5_000)
        lam = lambda_gc(p)
        assert 0.9 < lam < 1.1

    def test_inflation_high(self):
        # All tiny p-values → heavily inflated.
        lam = lambda_gc([1e-8] * 100)
        assert lam > 5.0

    def test_handles_bad_values(self):
        assert math.isnan(lambda_gc([]))
        assert math.isnan(lambda_gc([0.0, -1.0, float("nan"), 1.5]))

    def test_single_value_returns_nan(self):
        assert math.isnan(lambda_gc([0.5]))


# ---------------------------------------------------------------------------
# Top hits
# ---------------------------------------------------------------------------


class TestTopHits:
    def test_ranked_by_p_value(self):
        recs = _synthetic_records(
            n_per_chrom=20,
            planted=((("1", 150), 1e-10), (("2", 200), 1e-12)),
        )
        hits = top_hits(recs, n=3)
        assert hits[0]["variant_id"] == "2:200"
        assert hits[1]["variant_id"] == "1:150"
        assert len(hits) == 3

    def test_skips_invalid_p(self):
        recs = [
            {"chrom": "1", "pos": 1, "p_value": float("nan"),
             "variant_id": "a"},
            {"chrom": "1", "pos": 2, "p_value": 0.0, "variant_id": "b"},
            {"chrom": "1", "pos": 3, "p_value": 1.5, "variant_id": "c"},
            {"chrom": "1", "pos": 4, "p_value": 1e-6, "variant_id": "d"},
        ]
        hits = top_hits(recs, n=5)
        assert [h["variant_id"] for h in hits] == ["d"]


# ---------------------------------------------------------------------------
# Gene annotation
# ---------------------------------------------------------------------------


class TestGeneAnnotation:
    def test_load_gene_table_merges_transcripts(self, tmp_path):
        _write_fake_refgene(tmp_path)
        gt = load_gene_table("GRCh38", cache_dir=tmp_path, auto_download=False)
        assert set(gt["chr1"]) == {
            (100, 250, "GENEA", "+"),  # merged span of NM_001 + NM_001b
            (500, 800, "GENEB", "-"),
        }

    def test_auto_download_false_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_gene_table("GRCh38", cache_dir=tmp_path, auto_download=False)

    def test_load_gene_table_rejects_bad_build(self, tmp_path):
        with pytest.raises(ValueError):
            load_gene_table("NOT_A_BUILD", cache_dir=tmp_path)

    def test_annotate_hits_with_genes(self, tmp_path):
        _write_fake_refgene(tmp_path)
        gt = load_gene_table("GRCh38", cache_dir=tmp_path, auto_download=False)
        hits = [
            {"chrom": "1", "pos": 150},   # inside GENEA
            {"chrom": "1", "pos": 400},   # between GENEA and GENEB (closer to B)
            {"chrom": "chr2", "pos": 500},  # inside GENEC
            {"chrom": "3", "pos": 100},   # no gene on chr3
        ]
        anns = annotate_hits_with_genes(hits, gt, window_kb=1)
        assert anns[0].nearest_gene == "GENEA"
        assert anns[0].nearest_gene_distance_bp == 0
        assert anns[1].nearest_gene == "GENEB"
        assert anns[1].nearest_gene_distance_bp == 100
        assert anns[2].nearest_gene == "GENEC"
        assert anns[3].nearest_gene == ""
        assert anns[3].nearest_gene_distance_bp == -1

    def test_genes_in_region_strips_chr_prefix_either_way(self, tmp_path):
        _write_fake_refgene(tmp_path)
        gt = load_gene_table("GRCh38", cache_dir=tmp_path, auto_download=False)
        # chr1 entries stored as "chr1"; querying with "1" must still match.
        names = [g[2] for g in genes_in_region(gt, "1", 50, 1000)]
        assert names == ["GENEA", "GENEB"]

    def test_download_refuses_unsupported_build(self, tmp_path):
        with pytest.raises(ValueError):
            gwas_report.download_ucsc_refgene(
                "NONSENSE", cache_dir=tmp_path,
            )


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------


class TestFigures:
    def test_manhattan_figure_has_threshold_lines(self):
        recs = _synthetic_records(n_per_chrom=10)
        fig = build_manhattan_figure(recs, title="t")
        shapes = fig["layout"]["shapes"]
        assert len(shapes) == 2  # GWS + suggestive
        dashes = sorted(s["line"]["dash"] for s in shapes)
        assert dashes == ["dash", "dot"]
        assert fig["layout"]["title"]["text"].startswith("t")

    def test_manhattan_downsampling(self):
        # Many non-significant markers → non-sig points should be thinned.
        rng = np.random.default_rng(1)
        recs = [
            {"chrom": "1", "pos": i, "variant_id": f"v{i}",
             "p_value": float(rng.uniform(0.5, 1.0)),
             "beta": 0.0, "se": 1.0}
            for i in range(5000)
        ]
        fig = build_manhattan_figure(recs, title="t", max_nonsig_points=100)
        total_pts = sum(len(tr.get("x", [])) for tr in fig["data"])
        assert total_pts <= 150  # ~100 kept + small slack for sig/sug traces
        # The layout title notes the downsampling.
        assert "downsampled" in fig["layout"]["title"]["text"]

    def test_manhattan_keeps_all_significant_points(self):
        recs = _synthetic_records(
            n_per_chrom=500,
            planted=((("1", 500), 1e-10), (("2", 200), 1e-10)),
        )
        fig = build_manhattan_figure(recs, title="t", max_nonsig_points=100)
        # Locate the "genome-wide" trace and verify both points are present.
        sig_traces = [t for t in fig["data"]
                      if "genome-wide" in t.get("name", "")]
        assert len(sig_traces) == 1
        assert len(sig_traces[0]["x"]) == 2

    def test_qq_figure_structure(self):
        rng = np.random.default_rng(2)
        p = rng.uniform(0, 1, 5000)
        fig = build_qq_figure(p, title="q")
        title = fig["layout"]["title"]["text"]
        assert "λ" in title or "&lambda;" in title or "lambda" in title.lower() or "&#955;" in title
        # At least one scatter trace of observed points.
        assert any(t.get("mode") == "markers" for t in fig["data"])

    def test_qq_handles_empty(self):
        fig = build_qq_figure([], title="empty")
        assert fig["data"] == []

    def test_regional_figure_with_genes(self, tmp_path):
        _write_fake_refgene(tmp_path)
        gt = load_gene_table("GRCh38", cache_dir=tmp_path, auto_download=False)
        recs = [
            {"chrom": "1", "pos": p, "variant_id": f"v{p}",
             "p_value": 1e-9 if p == 150 else 0.5, "beta": 0.0, "se": 1.0}
            for p in range(100, 900, 10)
        ]
        fig = build_regional_figure(
            recs, chrom="1", center_pos=150,
            half_window_kb=1,  # 1 kb window
            genes=gt["chr1"],
            title="reg", lead_variant_id="v150",
        )
        # Gene shapes / annotations should be present.
        assert len(fig["layout"]["shapes"]) >= 1
        # Lead variant coloured red, larger marker.
        colors = fig["data"][0]["marker"]["color"]
        assert "#d62728" in colors


# ---------------------------------------------------------------------------
# Full report generation
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def test_report_runs_without_genes(self, tmp_path):
        p_auto = tmp_path / "auto.tsv"
        _write_results_tsv(p_auto, _synthetic_records(
            n_per_chrom=30,
            planted=((("1", 150), 1e-10),),
        ))
        out = tmp_path / "rpt.html"
        generate_gwas_report(
            {"autosomal": p_auto}, out,
            build=None, annotate_genes=False,
        )
        txt = out.read_text()
        assert out.exists() and out.stat().st_size > 1000
        for needle in ("Manhattan", "QQ", "Methods", "1:150"):
            assert needle in txt

    def test_report_all_sex_chr_modes(self, tmp_path):
        sources = {}
        for mode in [
            "autosomal",
            "x_with_sex_covariate",
            "x_male_only",
            "x_female_only",
            "y_male_only",
        ]:
            p = tmp_path / f"{mode}.tsv"
            chroms = ("1", "2") if mode == "autosomal" else (
                ("X",) if "x" in mode else ("Y",)
            )
            _write_results_tsv(p, _synthetic_records(
                n_per_chrom=15, chroms=chroms,
                planted=(((chroms[0], 150), 1e-9),),
            ))
            sources[mode] = p
        out = tmp_path / "rpt.html"
        generate_gwas_report(sources, out, annotate_genes=False)
        txt = out.read_text()
        # Each mode has its own section.
        for mode in sources:
            assert f'id="mode-{mode}"' in txt
        # Methods section + TOC.
        assert "<h2>Methods</h2>" in txt
        assert "<ul class='toc'>" in txt

    def test_report_with_gene_annotation(self, tmp_path):
        _write_fake_refgene(tmp_path / "cache")
        p = tmp_path / "auto.tsv"
        _write_results_tsv(p, _synthetic_records(
            n_per_chrom=30,
            chroms=("1",),
            planted=((("1", 150), 1e-10),),
        ))
        out = tmp_path / "rpt.html"
        generate_gwas_report(
            {"autosomal": p}, out,
            build="GRCh38", cache_dir=tmp_path / "cache",
            gene_window_kb=1,
        )
        txt = out.read_text()
        # The planted hit falls inside GENEA → must be annotated.
        assert "GENEA" in txt

    def test_report_writes_top_hits_tsv(self, tmp_path):
        p = tmp_path / "auto.tsv"
        _write_results_tsv(p, _synthetic_records(n_per_chrom=30))
        out = tmp_path / "rpt.html"
        generate_gwas_report(
            {"autosomal": p}, out, annotate_genes=False,
            top_hits_tsv_dir=tmp_path / "th",
        )
        th = tmp_path / "th" / "top_hits.autosomal.tsv"
        assert th.exists()
        with open(th, newline="") as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))
        assert 0 < len(rows) <= 10
        assert "variant_id" in rows[0]

    def test_report_empty_input_raises(self, tmp_path):
        empty = tmp_path / "e.tsv"
        empty.write_text(
            "chrom\tpos\tvariant_id\tbeta\tse\tstat\tp_value\tn_samples\tmethod\n"
        )
        out = tmp_path / "rpt.html"
        with pytest.raises(ValueError):
            generate_gwas_report(
                {"autosomal": empty}, out, annotate_genes=False,
            )

    def test_report_network_failure_falls_back_without_genes(
        self, tmp_path, monkeypatch,
    ):
        """When the UCSC download fails, the report is still produced."""
        def _boom(*_a, **_kw):
            raise RuntimeError("simulated network failure")
        monkeypatch.setattr(gwas_report, "download_ucsc_refgene", _boom)
        p = tmp_path / "auto.tsv"
        _write_results_tsv(p, _synthetic_records(n_per_chrom=10))
        out = tmp_path / "rpt.html"
        # Not cached and download raises → gene annotation silently disabled.
        generate_gwas_report(
            {"autosomal": p}, out,
            build="GRCh38", cache_dir=tmp_path / "nope",
        )
        assert out.exists()

    def test_summarize_mode_counts(self):
        recs = _synthetic_records(
            n_per_chrom=20,
            planted=(
                (("1", 150), 1e-10),  # GWS
                (("1", 160), 5e-6),   # suggestive
            ),
        )
        rep = summarize_mode("autosomal", recs, top_n=5)
        assert rep.n_tested == len(recs)
        assert rep.n_genome_wide == 1
        assert rep.n_suggestive == 2
        assert len(rep.top_hits) == 5


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestReportCli:
    def test_report_cli_generates_html(self, tmp_path):
        from array_lrr_gwas.cli import main

        p = tmp_path / "res.tsv"
        _write_results_tsv(p, _synthetic_records(n_per_chrom=25))
        out = tmp_path / "report.html"
        rc = main([
            "report",
            "--autosomal", str(p),
            "--no-gene-annotation",
            "-o", str(out),
        ])
        assert rc == 0
        assert out.exists() and out.stat().st_size > 1000

    def test_report_cli_missing_input_errors(self, tmp_path):
        from array_lrr_gwas.cli import main

        rc = main([
            "report",
            "-o", str(tmp_path / "out.html"),
            "--no-gene-annotation",
        ])
        assert rc == 1

    def test_report_cli_missing_file_errors(self, tmp_path):
        from array_lrr_gwas.cli import main

        rc = main([
            "report",
            "--autosomal", str(tmp_path / "nope.tsv"),
            "-o", str(tmp_path / "out.html"),
            "--no-gene-annotation",
        ])
        assert rc == 1


# ---------------------------------------------------------------------------
# Read/parse
# ---------------------------------------------------------------------------


class TestReadRecords:
    def test_round_trip(self, tmp_path):
        recs = _synthetic_records(n_per_chrom=5)
        p = tmp_path / "x.tsv"
        _write_results_tsv(p, recs)
        back = read_association_records(p)
        assert len(back) == len(recs)
        assert back[0]["chrom"] == recs[0]["chrom"]
        assert back[0]["pos"] == recs[0]["pos"]
        assert math.isclose(back[0]["p_value"], recs[0]["p_value"])
