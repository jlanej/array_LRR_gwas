"""Publication-quality GWAS summary report.

Generates a single self-contained interactive HTML report that collates
association results across autosomal and sex-chromosome analyses into a
reviewer-friendly document.  Content includes:

* Interactive Manhattan plots (per analysis mode) with gene annotations
  of nearby genes for genome-wide significant hits.
* Interactive QQ plots with 95 % confidence envelopes and the genomic
  inflation factor (lambda GC).
* Top-hit tables with nearest-gene and in-window gene annotations.
* Regional (locus-zoom style) plots for each significant locus.
* Methods narrative and figure legends suitable for inclusion in a
  manuscript's supplementary materials.

Gene annotations are sourced from UCSC's canonical ``refGene`` /
``ncbiRefSeq`` tracks for the appropriate genome build (GRCh37 → hg19,
GRCh38 → hg38, T2T-CHM13 → hs1).  Files are auto-downloaded on first
use and cached on disk; subsequent invocations reuse the cache.

Usage
-----
::

    from array_lrr_gwas.gwas_report import generate_gwas_report

    generate_gwas_report(
        {"autosomal": "results.tsv",
         "x_with_sex_covariate": "results.x_with_sex_covariate.tsv",
         ...},
        output_path="gwas_report.html",
        build="GRCh38",
        gene_window_kb=500,
    )

All plots are rendered with Plotly (already a package dependency) and
the resulting HTML is a self-contained document that can be opened in
any modern browser with no external resources beyond the Plotly CDN.
"""

from __future__ import annotations

import csv
import gzip
import html
import json
import logging
import math
import os
import re
import textwrap
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from array_lrr_gwas.genome_build import _normalise_build

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Suggestive / genome-wide significance thresholds (standard GWAS values).
GENOME_WIDE_P: float = 5e-8
SUGGESTIVE_P: float = 1e-5

# Default window (kb) for "near" gene annotation.
DEFAULT_GENE_WINDOW_KB: int = 500

# Default top-hits count for summary tables.
DEFAULT_TOP_N: int = 10

# Max number of non-suggestive points kept per analysis for the Manhattan
# plot.  Reduces HTML size while preserving the genome-wide picture.
_MAX_NONSIG_POINTS: int = 30_000

# Plotly CDN version pinned for reproducibility.
_PLOTLY_CDN: str = "https://cdn.plot.ly/plotly-2.35.2.min.js"

# UCSC refGene track URLs per build.  ncbiRefSeq for T2T-CHM13 (refGene is
# not populated for hs1; ncbiRefSeq is the canonical gene track).  These
# are the stable canonical public URLs and are hosted by UCSC.
_UCSC_GENE_URLS: dict[str, tuple[str, str]] = {
    # (url, table_name)
    "GRCh37": (
        "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/refGene.txt.gz",
        "refGene",
    ),
    "GRCh38": (
        "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/refGene.txt.gz",
        "refGene",
    ),
    "T2T-CHM13": (
        "https://hgdownload.soe.ucsc.edu/goldenPath/hs1/database/ncbiRefSeq.txt.gz",
        "ncbiRefSeq",
    ),
}

# Chromosome ordering for Manhattan plots (autosomes + X/Y/MT).
_CHROM_ORDER: list[str] = [str(i) for i in range(1, 23)] + ["X", "Y", "MT", "M"]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class GeneAnnotation:
    """Annotations attached to a single variant/hit."""

    nearest_gene: str = ""
    nearest_gene_distance_bp: int = -1  # -1 = no gene found
    genes_in_window: list[str] = field(default_factory=list)


@dataclass
class ModeReport:
    """Per-analysis-mode summary used to build the HTML report."""

    mode: str
    n_tested: int
    lambda_gc: float
    min_p: float
    n_genome_wide: int
    n_suggestive: int
    top_hits: list[dict]
    manhattan_fig: dict  # plotly figure dict
    qq_fig: dict         # plotly figure dict
    regional_figs: list[tuple[str, dict]]  # [(locus_label, fig_dict), ...]
    source_path: str = ""


# ---------------------------------------------------------------------------
# Genomic inflation factor
# ---------------------------------------------------------------------------


def lambda_gc(p_values: Iterable[float]) -> float:
    """Compute the genomic inflation factor (λ\ :sub:`GC`).

    λ\ :sub:`GC` is the ratio of the median observed χ²(1) statistic to
    the expected median (``0.4549364 ≈ qchisq(0.5, 1)``).  Well-controlled
    scans have ``λ ≈ 1``; values substantially above 1 indicate
    inflation from population stratification or cryptic relatedness.

    Parameters
    ----------
    p_values : iterable of float
        Association p-values.  Non-finite values and values ≤ 0 or > 1
        are discarded.

    Returns
    -------
    float
        ``lambda_gc``.  Returns ``nan`` when fewer than two usable
        p-values are provided.
    """
    try:
        from scipy.stats import chi2
    except ImportError:  # pragma: no cover - scipy is a hard dep
        chi2 = None

    arr = np.asarray(list(p_values), dtype=float)
    mask = np.isfinite(arr) & (arr > 0) & (arr <= 1.0)
    p = arr[mask]
    if p.size < 2:
        return float("nan")
    # Convert each p to a χ²(1) statistic via the inverse-CDF.
    if chi2 is not None:
        chi2_stats = chi2.isf(p, 1)
    else:  # pragma: no cover - fallback
        # Use a coarse approximation without scipy.
        chi2_stats = (np.sqrt(2.0) *
                      np.abs(_inv_norm_cdf(np.clip(p / 2.0, 1e-300, 0.5)))) ** 2
    med = float(np.median(chi2_stats))
    return med / 0.4549364


def _inv_norm_cdf(p: np.ndarray) -> np.ndarray:  # pragma: no cover
    """Minimal inverse-normal approximation (Acklam), fallback only."""
    p = np.asarray(p, dtype=float)
    # For the tails we use the asymptotic Mills ratio; adequate for fallback.
    return np.sqrt(-2.0 * np.log(p))


# ---------------------------------------------------------------------------
# Record I/O
# ---------------------------------------------------------------------------


def read_association_records(path: str | Path) -> list[dict]:
    """Read an association TSV written by ``array-lrr-gwas associate``.

    Returns a list of dicts with at least ``chrom``, ``pos``,
    ``variant_id``, ``beta``, ``se``, ``stat``, ``p_value`` keys cast to
    appropriate numeric types.  Any extra columns are preserved as
    strings.
    """
    path = Path(path)
    records: list[dict] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            try:
                pos = int(row["pos"])
            except (KeyError, ValueError, TypeError):
                continue  # skip malformed rows
            rec: dict = dict(row)
            rec["pos"] = pos
            for k in ("beta", "se", "stat", "p_value"):
                try:
                    rec[k] = float(row.get(k, "nan"))
                except (TypeError, ValueError):
                    rec[k] = float("nan")
            if "n_samples" in row:
                try:
                    rec["n_samples"] = int(row["n_samples"])
                except (TypeError, ValueError):
                    rec["n_samples"] = 0
            records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Top hits / summary table
# ---------------------------------------------------------------------------


def top_hits(
    records: Sequence[dict],
    n: int = DEFAULT_TOP_N,
    *,
    p_key: str = "p_value",
) -> list[dict]:
    """Return the top ``n`` records ranked by ascending ``p_value``.

    Non-finite or out-of-range p-values are ignored.  Ties are broken
    by ``chrom`` then ``pos`` for stable ordering.
    """
    usable = [
        r for r in records
        if np.isfinite(r.get(p_key, float("nan")))
        and 0.0 < r[p_key] <= 1.0
    ]
    usable.sort(
        key=lambda r: (r[p_key], _chrom_sort_key(r.get("chrom", "")), int(r.get("pos", 0)))
    )
    return usable[: max(0, int(n))]


def _chrom_sort_key(chrom: str) -> tuple[int, str]:
    """Sort key that puts 1..22 before X, Y, MT, then lexicographic."""
    c = str(chrom).replace("chr", "")
    try:
        return (int(c), "")
    except ValueError:
        order = {"X": 23, "Y": 24, "MT": 25, "M": 25}
        return (order.get(c, 99), c)


# ---------------------------------------------------------------------------
# UCSC gene download / caching
# ---------------------------------------------------------------------------


def _default_cache_dir() -> Path:
    """Return the default cache directory for UCSC gene tracks."""
    env = os.environ.get("ARRAY_LRR_GWAS_CACHE")
    if env:
        return Path(env)
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "array_lrr_gwas"


def download_ucsc_refgene(
    build: str,
    cache_dir: str | Path | None = None,
    *,
    force: bool = False,
    timeout: int = 60,
) -> Path:
    """Download the UCSC gene table for *build* to a local cache.

    Uses ``refGene.txt.gz`` for GRCh37/38 and ``ncbiRefSeq.txt.gz`` for
    T2T-CHM13 (hs1), which is the canonical gene track for that
    assembly at UCSC.  The file is kept as ``<build>_<table>.txt.gz``
    in the cache directory.

    Parameters
    ----------
    build : str
        Genome build name (any accepted by :func:`_normalise_build`).
    cache_dir : path-like, optional
        Override the default cache directory.
    force : bool
        Re-download even if the file exists.

    Returns
    -------
    Path
        Local path to the cached ``.txt.gz`` file.
    """
    canon = _normalise_build(build)
    if canon not in _UCSC_GENE_URLS:
        raise ValueError(
            f"Unsupported genome build for gene annotation: {build!r}. "
            f"Supported: {list(_UCSC_GENE_URLS)}"
        )
    url, table = _UCSC_GENE_URLS[canon]
    cdir = Path(cache_dir) if cache_dir else _default_cache_dir()
    cdir.mkdir(parents=True, exist_ok=True)
    target = cdir / f"{canon}_{table}.txt.gz"

    if target.exists() and not force and target.stat().st_size > 0:
        logger.info("UCSC %s cache hit: %s", table, target)
        return target

    logger.info("Downloading UCSC %s for %s from %s", table, canon, url)
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            # Protect against absurdly large payloads (sanity cap 500 MB).
            data = resp.read(500 * 1024 * 1024)
        tmp.write_bytes(data)
        tmp.replace(target)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise RuntimeError(
            f"Failed to download UCSC gene annotation for {canon} "
            f"from {url}: {exc}"
        ) from exc

    if not target.exists() or target.stat().st_size == 0:
        raise RuntimeError(
            f"Downloaded UCSC gene annotation for {canon} is empty."
        )
    return target


def load_gene_table(
    build: str,
    cache_dir: str | Path | None = None,
    *,
    auto_download: bool = True,
) -> dict[str, list[tuple[int, int, str, str]]]:
    """Load a gene table for *build*, indexed by chromosome.

    Returns a dict mapping ``chrom → sorted list of
    (tx_start, tx_end, gene_symbol, strand)`` tuples.  Coordinates are
    1-based inclusive (compatible with VCF POS).

    If the cached file does not exist and ``auto_download`` is True,
    the file is downloaded via :func:`download_ucsc_refgene`.
    """
    canon = _normalise_build(build)
    if canon not in _UCSC_GENE_URLS:
        raise ValueError(
            f"Unsupported genome build for gene annotation: {build!r}."
        )
    _, table = _UCSC_GENE_URLS[canon]
    cdir = Path(cache_dir) if cache_dir else _default_cache_dir()
    path = cdir / f"{canon}_{table}.txt.gz"
    if not path.exists() or path.stat().st_size == 0:
        if not auto_download:
            raise FileNotFoundError(
                f"Gene annotation for {canon} not cached at {path}. "
                "Pass auto_download=True or pre-populate the cache."
            )
        path = download_ucsc_refgene(canon, cache_dir=cdir)

    # refGene/ncbiRefSeq schema:
    # bin name chrom strand txStart txEnd cdsStart cdsEnd exonCount
    # exonStarts exonEnds score name2 cdsStartStat cdsEndStat exonFrames
    # UCSC txStart is 0-based, txEnd is 1-based exclusive => inclusive = txEnd.
    by_chrom: dict[str, dict[str, tuple[int, int, str]]] = {}
    n_rows = 0
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 13:
                continue
            chrom = fields[2]
            strand = fields[3]
            try:
                tx_start = int(fields[4]) + 1  # 0-based → 1-based
                tx_end = int(fields[5])        # exclusive → inclusive
            except ValueError:
                continue
            symbol = fields[12] or fields[1]  # name2, fallback to name
            if not symbol:
                continue
            # Aggregate by (chrom, symbol): take the widest span so that
            # a single gene with multiple transcripts is represented once.
            bucket = by_chrom.setdefault(chrom, {})
            cur = bucket.get(symbol)
            if cur is None:
                bucket[symbol] = (tx_start, tx_end, strand)
            else:
                bucket[symbol] = (
                    min(cur[0], tx_start),
                    max(cur[1], tx_end),
                    cur[2],
                )
            n_rows += 1

    result: dict[str, list[tuple[int, int, str, str]]] = {}
    for chrom, bucket in by_chrom.items():
        result[chrom] = sorted(
            (s, e, sym, strand) for sym, (s, e, strand) in bucket.items()
        )
    logger.info(
        "Loaded %d gene transcripts / %d unique genes for %s from %s",
        n_rows,
        sum(len(v) for v in result.values()),
        canon,
        path,
    )
    return result


# ---------------------------------------------------------------------------
# Nearby-gene annotation
# ---------------------------------------------------------------------------


def _norm_chrom(chrom: str) -> list[str]:
    """Return candidate chrom name variants (with/without chr prefix)."""
    c = str(chrom)
    if c.startswith("chr"):
        return [c, c[3:]]
    return [c, "chr" + c]


def annotate_hits_with_genes(
    hits: Sequence[dict],
    gene_table: dict[str, list[tuple[int, int, str, str]]],
    window_kb: int = DEFAULT_GENE_WINDOW_KB,
    *,
    max_in_window: int = 20,
) -> list[GeneAnnotation]:
    """Annotate each hit with the nearest gene and all genes in a window.

    Distance is 0 when the variant falls within a gene body; otherwise
    it is the base-pair distance to the closest gene boundary.

    Parameters
    ----------
    hits : sequence of dict
        Records with ``chrom`` and ``pos`` keys.
    gene_table : dict
        As returned by :func:`load_gene_table`.
    window_kb : int
        Half-window size (kb) used to collect "in-window" genes.
    max_in_window : int
        Cap the number of gene symbols in the in-window list per hit
        to keep the report compact.

    Returns
    -------
    list of GeneAnnotation
        Parallel to *hits*.
    """
    window_bp = int(window_kb) * 1000
    out: list[GeneAnnotation] = []
    for h in hits:
        ann = GeneAnnotation()
        pos = int(h.get("pos", 0))
        chrom_variants = _norm_chrom(h.get("chrom", ""))

        genes: list[tuple[int, int, str, str]] | None = None
        for cv in chrom_variants:
            if cv in gene_table:
                genes = gene_table[cv]
                break
        if not genes:
            out.append(ann)
            continue

        best_dist = None
        best_sym = ""
        in_window: list[tuple[int, str]] = []
        for (start, end, sym, _strand) in genes:
            # Distance (0 if inside)
            if start <= pos <= end:
                dist = 0
            elif pos < start:
                dist = start - pos
            else:
                dist = pos - end
            if dist <= window_bp:
                in_window.append((dist, sym))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_sym = sym
        if best_dist is not None:
            ann.nearest_gene = best_sym
            ann.nearest_gene_distance_bp = int(best_dist)
        in_window.sort()
        ann.genes_in_window = [s for _, s in in_window[:max_in_window]]
        out.append(ann)
    return out


def genes_in_region(
    gene_table: dict[str, list[tuple[int, int, str, str]]],
    chrom: str,
    start: int,
    end: int,
) -> list[tuple[int, int, str, str]]:
    """Return all genes overlapping ``chrom:start-end`` (1-based, inclusive)."""
    for cv in _norm_chrom(chrom):
        if cv in gene_table:
            return [
                (s, e, sym, strand)
                for (s, e, sym, strand) in gene_table[cv]
                if not (e < start or s > end)
            ]
    return []


# ---------------------------------------------------------------------------
# Genomic coordinate helpers
# ---------------------------------------------------------------------------


def _assign_cumulative_positions(
    records: Sequence[dict],
) -> tuple[list[str], dict[str, tuple[int, int]], np.ndarray]:
    """Compute per-chrom offsets for a single-axis Manhattan layout.

    Returns
    -------
    chroms : list[str]
        Ordered chromosome labels.
    offsets : dict[str, (offset, length)]
        Cumulative genomic offset and extent of each chromosome.
    cum_x : ndarray
        Parallel to ``records`` — cumulative x coordinate per variant.
    """
    # Determine max pos per chromosome
    max_pos: dict[str, int] = {}
    for r in records:
        c = str(r.get("chrom", "")).replace("chr", "")
        p = int(r.get("pos", 0))
        if p > max_pos.get(c, 0):
            max_pos[c] = p
    # Order chromosomes
    ordered = sorted(max_pos.keys(), key=_chrom_sort_key)
    offsets: dict[str, tuple[int, int]] = {}
    cum = 0
    for c in ordered:
        offsets[c] = (cum, max_pos[c])
        cum += max_pos[c] + 1_000_000  # 1 Mb gap between chromosomes
    cum_x = np.empty(len(records), dtype=np.int64)
    for i, r in enumerate(records):
        c = str(r.get("chrom", "")).replace("chr", "")
        cum_x[i] = offsets[c][0] + int(r.get("pos", 0))
    return ordered, offsets, cum_x


# ---------------------------------------------------------------------------
# Plot builders — return plotly figure *dicts* (pure data, no renderer)
# ---------------------------------------------------------------------------


def build_manhattan_figure(
    records: Sequence[dict],
    *,
    title: str,
    gene_labels: dict[int, str] | None = None,
    genome_wide_p: float = GENOME_WIDE_P,
    suggestive_p: float = SUGGESTIVE_P,
    max_nonsig_points: int = _MAX_NONSIG_POINTS,
    rng_seed: int = 0,
) -> dict:
    """Build a plotly figure dict for a Manhattan plot.

    Significant / suggestive hits are always retained; non-significant
    points are uniformly downsampled to ``max_nonsig_points`` to keep
    the HTML file size manageable for genome-wide scans.  The figure
    annotation notes this downsampling so readers can interpret it.

    Parameters
    ----------
    records : sequence of dict
        Association records (``chrom``, ``pos``, ``p_value`` at minimum).
    gene_labels : dict[int, str], optional
        Map from ``records`` index → gene-symbol label, used to
        annotate top hits directly on the plot.
    """
    rec_list = [r for r in records if np.isfinite(r.get("p_value", float("nan")))
                and 0.0 < r["p_value"] <= 1.0]
    n_total = len(rec_list)
    if n_total == 0:
        return _empty_fig(title, "Cumulative genomic position", "-log10(p)")

    ordered, offsets, cum_x = _assign_cumulative_positions(rec_list)
    p_vals = np.array([r["p_value"] for r in rec_list], dtype=float)
    log_p = -np.log10(np.maximum(p_vals, 1e-300))

    # Determine significance classes.
    sig_mask = p_vals < genome_wide_p
    sug_mask = (p_vals < suggestive_p) & ~sig_mask
    nonsig_mask = ~(sig_mask | sug_mask)

    # Downsample non-significant points.
    n_nonsig = int(nonsig_mask.sum())
    n_dropped = 0
    keep_mask = np.ones(n_total, dtype=bool)
    if n_nonsig > max_nonsig_points:
        rng = np.random.default_rng(rng_seed)
        nonsig_idx = np.where(nonsig_mask)[0]
        drop = rng.choice(nonsig_idx, size=n_nonsig - max_nonsig_points, replace=False)
        keep_mask[drop] = False
        n_dropped = int(len(drop))

    # Per-chromosome alternating colours for the non-significant points.
    chrom_colors = ["#1f77b4", "#2ca02c"]  # alternating blue / green
    traces: list[dict] = []
    for i, c in enumerate(ordered):
        col_mask = np.array(
            [str(r.get("chrom", "")).replace("chr", "") == c for r in rec_list],
            dtype=bool,
        )
        sel = col_mask & nonsig_mask & keep_mask
        if not sel.any():
            continue
        traces.append({
            "type": "scattergl",
            "x": cum_x[sel].tolist(),
            "y": log_p[sel].round(3).tolist(),
            "mode": "markers",
            "name": f"chr{c}",
            "legendgroup": "nonsig",
            "showlegend": False,
            "marker": {"size": 3, "color": chrom_colors[i % 2], "opacity": 0.7},
            "hoverinfo": "skip",
        })

    # Suggestive points (all kept).
    if sug_mask.any():
        traces.append({
            "type": "scattergl",
            "x": cum_x[sug_mask].tolist(),
            "y": log_p[sug_mask].round(4).tolist(),
            "mode": "markers",
            "name": f"suggestive (p<{suggestive_p:g})",
            "marker": {"size": 5, "color": "#ff7f0e"},
            "text": [
                _manhattan_hover(rec_list[i])
                for i in np.where(sug_mask)[0]
            ],
            "hoverinfo": "text",
        })

    # Genome-wide significant points.
    if sig_mask.any():
        sig_idx = np.where(sig_mask)[0]
        traces.append({
            "type": "scatter",
            "x": cum_x[sig_idx].tolist(),
            "y": log_p[sig_idx].round(4).tolist(),
            "mode": "markers",
            "name": f"genome-wide (p<{genome_wide_p:g})",
            "marker": {
                "size": 8,
                "color": "#d62728",
                "line": {"width": 0.5, "color": "#800000"},
            },
            "text": [_manhattan_hover(rec_list[i]) for i in sig_idx],
            "hoverinfo": "text",
        })

    # Gene annotations (top hits only).
    annotations = []
    if gene_labels:
        for idx, sym in gene_labels.items():
            if not sym or idx >= n_total:
                continue
            annotations.append({
                "x": int(cum_x[idx]),
                "y": float(log_p[idx]),
                "text": sym,
                "showarrow": True,
                "arrowhead": 2,
                "arrowsize": 0.8,
                "ax": 0,
                "ay": -25,
                "font": {"size": 10, "color": "#333"},
            })

    # X-axis ticks: chromosome centres.
    tickvals = []
    ticktext = []
    for c in ordered:
        off, length = offsets[c]
        tickvals.append(off + length // 2)
        ticktext.append(c)

    shapes = [
        # Genome-wide significance line
        {
            "type": "line", "x0": 0, "x1": 1, "xref": "paper",
            "y0": -math.log10(genome_wide_p), "y1": -math.log10(genome_wide_p),
            "line": {"color": "#d62728", "dash": "dash", "width": 1},
        },
        # Suggestive significance line
        {
            "type": "line", "x0": 0, "x1": 1, "xref": "paper",
            "y0": -math.log10(suggestive_p), "y1": -math.log10(suggestive_p),
            "line": {"color": "#ff7f0e", "dash": "dot", "width": 1},
        },
    ]

    subtitle = ""
    if n_dropped > 0:
        subtitle = (
            f"<br><span style='font-size:11px;color:#666'>Showing "
            f"{n_total - n_dropped:,} / {n_total:,} markers "
            f"(non-significant points downsampled to {max_nonsig_points:,}).</span>"
        )

    layout = {
        "title": {"text": title + subtitle, "x": 0.02, "xanchor": "left"},
        "xaxis": {
            "title": "Chromosome",
            "tickmode": "array",
            "tickvals": tickvals,
            "ticktext": ticktext,
            "showgrid": False,
            "zeroline": False,
        },
        "yaxis": {
            "title": "-log<sub>10</sub>(p)",
            "zeroline": False,
            "rangemode": "tozero",
        },
        "hovermode": "closest",
        "showlegend": True,
        "legend": {"orientation": "h", "y": 1.08, "x": 1, "xanchor": "right"},
        "shapes": shapes,
        "annotations": annotations,
        "margin": {"l": 60, "r": 20, "t": 70, "b": 60},
    }
    return {"data": traces, "layout": layout}


def _manhattan_hover(rec: dict) -> str:
    vid = rec.get("variant_id") or rec.get("id") or f"{rec.get('chrom')}:{rec.get('pos')}"
    p = rec.get("p_value", float("nan"))
    beta = rec.get("beta", float("nan"))
    se = rec.get("se", float("nan"))
    return (
        f"{vid}<br>chr{str(rec.get('chrom', '')).replace('chr', '')}:"
        f"{rec.get('pos', '?')}<br>"
        f"p = {p:.3g}<br>β = {beta:.3g} ± {se:.3g}"
    )


def build_qq_figure(
    p_values: Iterable[float],
    *,
    title: str,
    max_points: int = 50_000,
    ci_alpha: float = 0.95,
) -> dict:
    """Build a plotly QQ-plot figure dict with a 95 % confidence band.

    All significant points (top 1000 by p-value) are always retained;
    the remainder are log-uniformly thinned to ``max_points`` total.
    """
    arr = np.asarray(list(p_values), dtype=float)
    mask = np.isfinite(arr) & (arr > 0) & (arr <= 1.0)
    p = np.sort(arr[mask])
    n = p.size
    if n == 0:
        return _empty_fig(title, "Expected -log10(p)", "Observed -log10(p)")

    ranks = np.arange(1, n + 1)
    exp_p = ranks / (n + 1.0)
    exp_log = -np.log10(exp_p)
    obs_log = -np.log10(np.maximum(p, 1e-300))

    # Thin to max_points while preserving the tail.
    keep = np.ones(n, dtype=bool)
    if n > max_points:
        tail_n = min(1000, n)
        # always keep the smallest p-values (tail of QQ)
        thin = np.linspace(tail_n, n - 1, max_points - tail_n, dtype=int)
        mask2 = np.zeros(n, dtype=bool)
        mask2[:tail_n] = True
        mask2[thin] = True
        keep = mask2

    # 95 % pointwise CI under the null via Beta(k, n-k+1) quantiles.
    try:
        from scipy.stats import beta as beta_dist
        alpha = 1.0 - ci_alpha
        lo = beta_dist.ppf(alpha / 2, ranks[keep], n - ranks[keep] + 1)
        hi = beta_dist.ppf(1 - alpha / 2, ranks[keep], n - ranks[keep] + 1)
        ci_lo = -np.log10(np.maximum(lo, 1e-300))
        ci_hi = -np.log10(np.maximum(hi, 1e-300))
    except ImportError:  # pragma: no cover
        ci_lo = ci_hi = None

    lam = lambda_gc(p)

    traces: list[dict] = []
    # Null band
    if ci_lo is not None and ci_hi is not None:
        xs = exp_log[keep].tolist()
        traces.append({
            "type": "scatter",
            "x": xs + xs[::-1],
            "y": ci_hi.tolist() + ci_lo.tolist()[::-1],
            "fill": "toself",
            "fillcolor": "rgba(150,150,150,0.25)",
            "line": {"color": "rgba(0,0,0,0)"},
            "name": f"{int(ci_alpha * 100)}% CI under null",
            "hoverinfo": "skip",
            "showlegend": True,
        })

    # y = x reference line
    lim = float(max(exp_log.max(), obs_log.max()))
    traces.append({
        "type": "scatter",
        "x": [0, lim],
        "y": [0, lim],
        "mode": "lines",
        "name": "y = x",
        "line": {"dash": "dash", "color": "#444"},
        "hoverinfo": "skip",
    })

    # Observed QQ points
    traces.append({
        "type": "scattergl",
        "x": exp_log[keep].round(3).tolist(),
        "y": obs_log[keep].round(4).tolist(),
        "mode": "markers",
        "name": "observed",
        "marker": {"size": 4, "color": "#1f77b4"},
        "hoverinfo": "x+y",
    })

    layout = {
        "title": {
            "text": f"{title}<br><span style='font-size:12px;color:#666'>"
                    f"λ<sub>GC</sub> = {lam:.3f} (n = {n:,} tested)</span>",
            "x": 0.02, "xanchor": "left",
        },
        "xaxis": {"title": "Expected -log<sub>10</sub>(p)", "zeroline": False,
                  "range": [0, lim * 1.02]},
        "yaxis": {"title": "Observed -log<sub>10</sub>(p)", "zeroline": False,
                  "range": [0, lim * 1.02]},
        "hovermode": "closest",
        "margin": {"l": 60, "r": 20, "t": 70, "b": 60},
    }
    return {"data": traces, "layout": layout}


def build_regional_figure(
    records: Sequence[dict],
    *,
    chrom: str,
    center_pos: int,
    half_window_kb: int = 500,
    genes: Sequence[tuple[int, int, str, str]] = (),
    title: str = "",
    lead_variant_id: str | None = None,
) -> dict:
    """Build a regional (locus-zoom style) plotly figure dict.

    Plots ``-log10(p)`` for all records within ±``half_window_kb``
    of ``center_pos`` on ``chrom``; a second subplot (stacked vertically)
    shows gene track positions with strand-aware triangles.
    """
    window_bp = int(half_window_kb) * 1000
    lo = center_pos - window_bp
    hi = center_pos + window_bp

    sel = [
        r for r in records
        if str(r.get("chrom", "")).replace("chr", "") ==
        str(chrom).replace("chr", "")
        and lo <= int(r.get("pos", -1)) <= hi
        and np.isfinite(r.get("p_value", float("nan")))
        and 0.0 < r["p_value"] <= 1.0
    ]
    if not sel:
        return _empty_fig(title, f"Position on chr{chrom} (bp)", "-log10(p)")

    x = np.array([int(r["pos"]) for r in sel], dtype=np.int64)
    y = -np.log10(np.maximum(np.array([r["p_value"] for r in sel]), 1e-300))

    hover = [_manhattan_hover(r) for r in sel]
    colors = [
        "#d62728" if (lead_variant_id and r.get("variant_id") == lead_variant_id)
        else "#1f77b4"
        for r in sel
    ]
    sizes = [
        10 if (lead_variant_id and r.get("variant_id") == lead_variant_id) else 5
        for r in sel
    ]

    traces: list[dict] = [{
        "type": "scatter",
        "x": x.tolist(),
        "y": y.round(4).tolist(),
        "mode": "markers",
        "name": "markers",
        "marker": {"size": sizes, "color": colors, "line": {"width": 0.3, "color": "#222"}},
        "text": hover,
        "hoverinfo": "text",
        "xaxis": "x",
        "yaxis": "y",
    }]

    # Gene track as shapes / markers on a second y-axis.
    gene_annots: list[dict] = []
    gene_shapes: list[dict] = []
    for i, (gs, ge, sym, strand) in enumerate(genes):
        # Only draw genes whose span intersects the window.
        xs = max(gs, lo)
        xe = min(ge, hi)
        if xe < xs:
            continue
        y_pos = 0.5 + (i % 3) * 0.5  # stagger to reduce overlap
        gene_shapes.append({
            "type": "rect",
            "x0": xs, "x1": xe,
            "y0": y_pos - 0.15, "y1": y_pos + 0.15,
            "xref": "x2", "yref": "y2",
            "fillcolor": "#555",
            "line": {"width": 0},
            "opacity": 0.85,
        })
        gene_annots.append({
            "x": (xs + xe) / 2,
            "y": y_pos + 0.25,
            "xref": "x2", "yref": "y2",
            "text": f"<i>{html.escape(sym)}</i> " + (
                "→" if strand == "+" else ("←" if strand == "-" else "")
            ),
            "showarrow": False,
            "font": {"size": 11, "color": "#222"},
        })

    layout = {
        "title": {"text": title or f"Regional plot chr{chrom}:{center_pos:,}",
                  "x": 0.02, "xanchor": "left"},
        "xaxis": {
            "title": "",
            "domain": [0, 1],
            "anchor": "y",
            "range": [lo, hi],
            "showticklabels": False,
        },
        "yaxis": {
            "title": "-log<sub>10</sub>(p)",
            "domain": [0.35, 1.0],
            "anchor": "x",
        },
        "xaxis2": {
            "title": f"Position on chr{chrom} (bp)",
            "domain": [0, 1],
            "anchor": "y2",
            "range": [lo, hi],
            "matches": "x",
        },
        "yaxis2": {
            "title": "Genes",
            "domain": [0, 0.3],
            "anchor": "x2",
            "range": [0, 3.0],
            "showticklabels": False,
            "zeroline": False,
        },
        "shapes": gene_shapes,
        "annotations": gene_annots,
        "showlegend": False,
        "hovermode": "closest",
        "margin": {"l": 60, "r": 20, "t": 60, "b": 60},
    }
    return {"data": traces, "layout": layout}


def _empty_fig(title: str, xlab: str, ylab: str) -> dict:
    return {
        "data": [],
        "layout": {
            "title": {"text": title + "<br><span style='font-size:12px;color:#888'>"
                      "(no data)</span>", "x": 0.02, "xanchor": "left"},
            "xaxis": {"title": xlab},
            "yaxis": {"title": ylab},
            "margin": {"l": 60, "r": 20, "t": 70, "b": 60},
        },
    }


# ---------------------------------------------------------------------------
# Per-mode summary assembly
# ---------------------------------------------------------------------------


def summarize_mode(
    mode: str,
    records: Sequence[dict],
    *,
    source_path: str = "",
    gene_table: dict | None = None,
    gene_window_kb: int = DEFAULT_GENE_WINDOW_KB,
    top_n: int = DEFAULT_TOP_N,
    regional_half_window_kb: int = 500,
    max_regional_loci: int = 5,
) -> ModeReport:
    """Produce a :class:`ModeReport` from per-marker association records."""
    valid_p = [
        r["p_value"] for r in records
        if np.isfinite(r.get("p_value", float("nan")))
        and 0.0 < r["p_value"] <= 1.0
    ]
    lam = lambda_gc(valid_p)
    min_p = float(min(valid_p)) if valid_p else float("nan")
    n_gws = int(sum(1 for p in valid_p if p < GENOME_WIDE_P))
    n_sug = int(sum(1 for p in valid_p if p < SUGGESTIVE_P))
    n_tested = len(valid_p)

    hits = top_hits(records, n=top_n)

    # Gene annotation for hits.
    gene_labels_for_manhattan: dict[int, str] = {}
    if gene_table:
        anns = annotate_hits_with_genes(hits, gene_table, window_kb=gene_window_kb)
        # Attach to hit records (non-destructive to input).
        hits_out: list[dict] = []
        for h, ann in zip(hits, anns):
            hc = dict(h)
            hc["nearest_gene"] = ann.nearest_gene
            hc["nearest_gene_distance_bp"] = ann.nearest_gene_distance_bp
            hc["genes_in_window"] = ",".join(ann.genes_in_window)
            hits_out.append(hc)
        hits = hits_out
        # Build manhattan gene-label mapping (only genome-wide-significant hits).
        # We need the index within the full records list for placement.
        idx_by_key: dict[tuple[str, int, str], int] = {
            (str(r.get("chrom", "")), int(r.get("pos", 0)),
             str(r.get("variant_id", ""))): i
            for i, r in enumerate(records)
        }
        for h in hits:
            if h.get("p_value", 1) >= GENOME_WIDE_P:
                continue
            key = (str(h.get("chrom", "")), int(h.get("pos", 0)),
                   str(h.get("variant_id", "")))
            idx = idx_by_key.get(key)
            if idx is not None and h.get("nearest_gene"):
                gene_labels_for_manhattan[idx] = h["nearest_gene"]
    else:
        hits_out = []
        for h in hits:
            hc = dict(h)
            hc.setdefault("nearest_gene", "")
            hc.setdefault("nearest_gene_distance_bp", -1)
            hc.setdefault("genes_in_window", "")
            hits_out.append(hc)
        hits = hits_out

    manhattan = build_manhattan_figure(
        records,
        title=f"Manhattan — {mode}",
        gene_labels=gene_labels_for_manhattan,
    )
    qq = build_qq_figure(
        [r.get("p_value", float("nan")) for r in records],
        title=f"QQ — {mode}",
    )

    # Regional figures for independent genome-wide-significant loci
    # (collapse lead hits within ±regional_half_window_kb of each other).
    regional_figs: list[tuple[str, dict]] = []
    seen: list[tuple[str, int]] = []
    window_bp = regional_half_window_kb * 1000
    for h in hits:
        if len(regional_figs) >= max_regional_loci:
            break
        if not (0 < h.get("p_value", 1) < SUGGESTIVE_P):
            continue
        chrom = str(h.get("chrom", "")).replace("chr", "")
        pos = int(h.get("pos", 0))
        if any(c == chrom and abs(pos - p) < window_bp for c, p in seen):
            continue
        seen.append((chrom, pos))
        region_genes = (
            genes_in_region(
                gene_table, chrom,
                max(0, pos - window_bp), pos + window_bp,
            )
            if gene_table else []
        )
        label = (
            f"chr{chrom}:{pos:,} "
            + (f"({h['nearest_gene']})" if h.get("nearest_gene") else "")
        ).strip()
        regional_figs.append((
            label,
            build_regional_figure(
                records,
                chrom=chrom,
                center_pos=pos,
                half_window_kb=regional_half_window_kb,
                genes=region_genes,
                title=f"Regional plot — {label}",
                lead_variant_id=h.get("variant_id"),
            ),
        ))

    return ModeReport(
        mode=mode,
        n_tested=n_tested,
        lambda_gc=lam,
        min_p=min_p,
        n_genome_wide=n_gws,
        n_suggestive=n_sug,
        top_hits=hits,
        manhattan_fig=manhattan,
        qq_fig=qq,
        regional_figs=regional_figs,
        source_path=source_path,
    )


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


def _fig_to_json(fig: dict) -> str:
    return json.dumps(fig, default=_json_default, allow_nan=False)


def _format_p(p: float) -> str:
    if not np.isfinite(p) or p <= 0:
        return "—"
    if p >= 1e-3:
        return f"{p:.3g}"
    return f"{p:.2e}"


def _hits_table_html(hits: Sequence[dict], *, with_gene_cols: bool) -> str:
    if not hits:
        return '<p class="muted">No hits to display.</p>'
    # Columns
    base_cols = [
        ("variant_id", "Variant"),
        ("chrom", "Chr"),
        ("pos", "Position"),
        ("beta", "β"),
        ("se", "SE"),
        ("p_value", "p"),
    ]
    extra_cols = []
    if "n_samples" in hits[0]:
        extra_cols.append(("n_samples", "N"))
    if with_gene_cols:
        extra_cols.append(("nearest_gene", "Nearest gene"))
        extra_cols.append(("nearest_gene_distance_bp", "Distance (bp)"))
        extra_cols.append(("genes_in_window", "Genes in window"))

    cols = base_cols + extra_cols
    out = ['<table class="hits-table"><thead><tr>']
    for _, label in cols:
        out.append(f"<th>{html.escape(label)}</th>")
    out.append("</tr></thead><tbody>")
    for h in hits:
        out.append("<tr>")
        for key, _ in cols:
            val = h.get(key, "")
            if key == "p_value":
                txt = _format_p(float(val) if val != "" else float("nan"))
            elif key in ("beta", "se"):
                try:
                    txt = f"{float(val):.3g}"
                except (TypeError, ValueError):
                    txt = str(val)
            elif key == "nearest_gene_distance_bp":
                try:
                    d = int(val)
                    txt = "—" if d < 0 else f"{d:,}"
                except (TypeError, ValueError):
                    txt = "—"
            elif key == "pos":
                try:
                    txt = f"{int(val):,}"
                except (TypeError, ValueError):
                    txt = str(val)
            else:
                txt = str(val)
            out.append(f"<td>{html.escape(txt)}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def _methods_html(
    *,
    build: str | None,
    gene_window_kb: int,
    gene_source: str | None,
    genome_wide_p: float,
    suggestive_p: float,
) -> str:
    """Return a short methods / interpretation section (HTML)."""
    build_txt = (
        html.escape(build) if build else "not specified"
    )
    source_txt = (
        f"UCSC <code>{html.escape(gene_source)}</code>"
        if gene_source else "not annotated"
    )
    return textwrap.dedent(f"""\
    <h2>Methods</h2>
    <div class="info">
      <p>
        Association statistics were produced by
        <code>array-lrr-gwas associate</code>, which models Log R Ratio
        (LRR) as a quantitative proxy for copy-number dosage at each
        array marker.  For autosomal scans, a linear mixed model (LMM)
        is fitted with a genetic relatedness matrix (GRM) estimated
        from LD-pruned markers (PLINK2 by default), falling back to
        ordinary least squares (OLS) when requested.  chrX scans use a
        dedicated X-chromosome GRM (X-GRM) with GCTA-style male 0/2
        dosage coding and pseudoautosomal region (PAR) exclusion;
        sex-stratified modes (<code>x_male_only</code>,
        <code>x_female_only</code>, <code>y_male_only</code>,
        <code>mt_male_only</code>, <code>mt_female_only</code>)
        additionally drop the sex covariate to avoid singularity.
        chrM/MT scans reuse the autosomal GRM as a practical adjuster
        for cohort structure and shared technical batch effects; this
        is not a strict mitochondrial kinship matrix (mtDNA is
        maternally inherited and does not recombine), and the
        <code>mt_female_only</code> mode is the most biologically
        direct, representing the maternal transmission lineage.
      </p>
      <p>
        <strong>Genomic inflation.</strong> The genomic inflation
        factor λ<sub>GC</sub> is computed from the median of the
        χ²(1) statistics derived from each p-value
        (<code>λ = median(χ²) / 0.4549</code>).  Values close to 1
        indicate appropriate control for confounding; values ≫ 1
        suggest residual stratification or cryptic relatedness.
      </p>
      <p>
        <strong>Significance thresholds.</strong> The
        dashed red line on each Manhattan plot marks the genome-wide
        significance threshold (p&nbsp;&lt;&nbsp;{genome_wide_p:g});
        the dotted orange line marks the suggestive threshold
        (p&nbsp;&lt;&nbsp;{suggestive_p:g}).
      </p>
      <p>
        <strong>Gene annotation.</strong> Hits are annotated with the
        nearest gene and all genes falling within a
        ±{gene_window_kb}&nbsp;kb window around each lead marker.
        Gene coordinates are taken from {source_txt} on
        <strong>{build_txt}</strong> and cached locally on first use.
      </p>
      <p>
        <strong>Downsampling.</strong> For file-size reasons,
        non-significant markers may be uniformly downsampled in the
        Manhattan plot; every marker with p&nbsp;&lt;&nbsp;{suggestive_p:g}
        is always retained.  All summary statistics (λ<sub>GC</sub>,
        hit counts, top-hit tables) use the complete (un-downsampled)
        result set.
      </p>
    </div>
    """)


def _mode_html(mode_key: str, report: ModeReport, fig_counter: list[int]) -> str:
    """Render a single <section> for one analysis mode."""
    # Pretty mode label
    pretty = {
        "autosomal": "Autosomal",
        "x_with_sex_covariate": "chrX — full cohort (sex as covariate)",
        "x_male_only": "chrX — males only",
        "x_female_only": "chrX — females only",
        "y_male_only": "chrY — males only",
        "mt_with_sex_covariate": "chrM/MT — full cohort (sex as covariate)",
        "mt_male_only": "chrM/MT — males only",
        "mt_female_only": "chrM/MT — females only (maternal lineage)",
    }.get(mode_key, mode_key)

    # Headline summary
    summary = textwrap.dedent(f"""\
    <div class="info">
      <strong>{html.escape(pretty)}</strong>
      &middot; Source: <code>{html.escape(report.source_path or "—")}</code><br>
      Markers tested: <strong>{report.n_tested:,}</strong>
      &middot; λ<sub>GC</sub>: <strong>{report.lambda_gc:.3f}</strong>
      &middot; Min p: <strong>{_format_p(report.min_p)}</strong>
      &middot; Genome-wide (p&lt;{GENOME_WIDE_P:g}):
        <strong>{report.n_genome_wide}</strong>
      &middot; Suggestive (p&lt;{SUGGESTIVE_P:g}):
        <strong>{report.n_suggestive}</strong>
    </div>
    """)

    man_id = f"fig-{fig_counter[0]}"; fig_counter[0] += 1
    qq_id = f"fig-{fig_counter[0]}"; fig_counter[0] += 1

    regional_blocks = []
    if report.regional_figs:
        regional_blocks.append('<h3 class="sub">Regional plots (top loci)</h3>')
        for label, fig in report.regional_figs:
            fid = f"fig-{fig_counter[0]}"; fig_counter[0] += 1
            regional_blocks.append(
                f'<div class="plot-container" id="{fid}"></div>'
                f'<script type="application/json" data-figure-id="{fid}">'
                f"{_fig_to_json(fig)}</script>"
            )

    with_gene_cols = any("nearest_gene" in h for h in report.top_hits)
    hits_html = _hits_table_html(report.top_hits, with_gene_cols=with_gene_cols)

    lam_comment = _interpret_lambda(report.lambda_gc)

    return textwrap.dedent(f"""\
    <section class="mode-section" id="mode-{html.escape(mode_key)}">
      <h2>{html.escape(pretty)}</h2>
      {summary}
      <p class="muted">{lam_comment}</p>

      <h3 class="sub">Manhattan plot</h3>
      <div class="plot-container" id="{man_id}"></div>
      <script type="application/json" data-figure-id="{man_id}">{_fig_to_json(report.manhattan_fig)}</script>

      <h3 class="sub">QQ plot</h3>
      <div class="plot-container" id="{qq_id}"></div>
      <script type="application/json" data-figure-id="{qq_id}">{_fig_to_json(report.qq_fig)}</script>

      {"".join(regional_blocks)}

      <h3 class="sub">Top {len(report.top_hits)} hits</h3>
      {hits_html}
    </section>
    """)


def _interpret_lambda(lam: float) -> str:
    if not np.isfinite(lam):
        return "λ<sub>GC</sub> could not be computed (no valid p-values)."
    if lam < 1.02:
        tag = "well-controlled"
    elif lam < 1.05:
        tag = "acceptable (minor inflation)"
    elif lam < 1.10:
        tag = "moderate inflation — consider additional covariate adjustment"
    else:
        tag = (
            "substantial inflation — residual population structure or "
            "cryptic relatedness likely; interpret hits cautiously"
        )
    return f"Genomic inflation {tag} (λ<sub>GC</sub> = {lam:.3f})."


def _toc_html(report_by_mode: dict[str, ModeReport]) -> str:
    items = []
    for k, r in report_by_mode.items():
        items.append(
            f'<li><a href="#mode-{html.escape(k)}">{html.escape(k)}</a> '
            f'&middot; n={r.n_tested:,} &middot; λ={r.lambda_gc:.3f} '
            f'&middot; gws={r.n_genome_wide}</li>'
        )
    return "<ul class='toc'>" + "".join(items) + "</ul>"


def _render_html(
    *,
    report_by_mode: dict[str, ModeReport],
    title: str,
    build: str | None,
    gene_window_kb: int,
    gene_source: str | None,
    combined_manhattan_fig: dict | None = None,
    summary_modes: Sequence[str] | None = None,
) -> str:
    fig_counter = [0]
    mode_blocks = "\n".join(
        _mode_html(k, v, fig_counter) for k, v in report_by_mode.items()
    )
    toc = _toc_html(report_by_mode)
    methods = _methods_html(
        build=build,
        gene_window_kb=gene_window_kb,
        gene_source=gene_source,
        genome_wide_p=GENOME_WIDE_P,
        suggestive_p=SUGGESTIVE_P,
    )

    # Optional combined summary Manhattan block (shown above the
    # per-mode sections) overlaying autosomal + headline sex/MT scans.
    summary_block = ""
    if combined_manhattan_fig is not None:
        fid = f"fig-summary-{fig_counter[0]}"; fig_counter[0] += 1
        modes_txt = (
            html.escape(", ".join(summary_modes))
            if summary_modes else ""
        )
        summary_block = textwrap.dedent(f"""\
        <section class="mode-section" id="mode-summary">
          <h2>Genome-wide summary</h2>
          <div class="info">
            Combined Manhattan plot overlaying the autosomal scan with the
            headline non-autosomal scans ({modes_txt}).  chrX uses the
            sex-as-covariate mode, chrY uses the male-only mode, and
            chrM/MT uses the sex-as-covariate mode.  Per-mode Manhattan,
            QQ, regional, and top-hit views are available in the sections
            below.
          </div>
          <div class="plot-container" id="{fid}"></div>
          <script type="application/json" data-figure-id="{fid}">{_fig_to_json(combined_manhattan_fig)}</script>
        </section>
        """)

    title_e = html.escape(title)
    build_e = html.escape(build or "unknown")

    # Use a safe JS template that reads each figure from its <script
    # type="application/json"> sibling.  This avoids embedding a single
    # huge JSON blob and keeps per-figure payload scoped.
    render_js = r"""
    (function() {
      if (typeof Plotly === 'undefined') {
        console.error('Plotly did not load');
        return;
      }
      var nodes = document.querySelectorAll('script[type="application/json"][data-figure-id]');
      nodes.forEach(function(node) {
        try {
          var fig = JSON.parse(node.textContent);
          var id = node.getAttribute('data-figure-id');
          Plotly.newPlot(id, fig.data || [], fig.layout || {}, {
            responsive: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['lasso2d', 'select2d']
          });
        } catch (err) {
          console.error('Failed to render figure', err);
        }
      });
    })();
    """

    html_doc = textwrap.dedent(f"""\
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{title_e}</title>
    <script src="{_PLOTLY_CDN}" charset="utf-8"></script>
    <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                     "Helvetica Neue", Arial, sans-serif;
        background: #f7f8fa; color: #222; line-height: 1.55;
    }}
    .container {{ max-width: 1280px; margin: 0 auto; padding: 20px; }}
    h1 {{ font-size: 1.9rem; margin-bottom: 8px; color: #1a1a2e; }}
    h2 {{ font-size: 1.35rem; margin: 28px 0 10px; color: #16213e;
         border-bottom: 2px solid #0f3460; padding-bottom: 4px; }}
    h3.sub {{ font-size: 1.05rem; margin: 18px 0 8px; color: #333; }}
    .info {{ background: #e8f4f8; border-left: 4px solid #0f3460;
            padding: 10px 14px; margin: 10px 0; border-radius: 4px;
            font-size: 0.93rem; }}
    .muted {{ color: #555; font-size: 0.9rem; margin: 6px 0; }}
    .plot-container {{ background: #fff; border: 1px solid #ddd;
                      border-radius: 6px; padding: 8px; margin: 8px 0;
                      min-height: 420px; }}
    table.hits-table {{ border-collapse: collapse; width: 100%;
                       background: #fff; font-size: 0.9rem; margin: 8px 0;
                       border: 1px solid #ddd; }}
    table.hits-table th, table.hits-table td {{ padding: 6px 10px;
        border-bottom: 1px solid #eee; text-align: left; }}
    table.hits-table th {{ background: #f0f3f7; color: #16213e;
        border-bottom: 2px solid #0f3460; }}
    table.hits-table tr:nth-child(even) td {{ background: #fafbfc; }}
    .toc {{ list-style: none; padding: 8px 16px; background: #fff;
           border: 1px solid #ddd; border-radius: 6px; margin: 12px 0; }}
    .toc li {{ padding: 4px 0; font-size: 0.93rem; }}
    footer {{ margin-top: 32px; padding: 16px 0; border-top: 1px solid #ddd;
             font-size: 0.85rem; color: #666; text-align: center; }}
    code {{ background: #eef2f6; padding: 1px 5px; border-radius: 3px;
           font-size: 0.85em; }}
    </style>
    </head>
    <body>
    <div class="container">
      <h1>{title_e}</h1>
      <div class="info">
        Publication-quality summary of array-LRR GWAS association
        results.  Reference build: <strong>{build_e}</strong>.
        All Manhattan, QQ, and regional plots below are interactive
        (hover, zoom, pan); use the toolbar in the top-right of each
        plot to reset or save as PNG.
      </div>
      <h2>Contents</h2>
      {toc}
      {summary_block}
      {mode_blocks}
      {methods}
      <footer>
        Generated by <code>array-lrr-gwas report</code> &mdash;
        <a href="https://github.com/jlanej/array_LRR_gwas">array_LRR_gwas</a>
      </footer>
    </div>
    <script>{render_js}</script>
    </body>
    </html>
    """)
    return html_doc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_gwas_report(
    mode_sources: dict[str, str | Path | Sequence[dict]],
    output_path: str | Path,
    *,
    build: str | None = None,
    gene_window_kb: int = DEFAULT_GENE_WINDOW_KB,
    top_n: int = DEFAULT_TOP_N,
    cache_dir: str | Path | None = None,
    title: str = "Array-LRR GWAS — Summary Report",
    annotate_genes: bool = True,
    regional_half_window_kb: int = 500,
    max_regional_loci: int = 5,
    top_hits_tsv_dir: str | Path | None = None,
) -> Path:
    """Build and write a single-file interactive HTML GWAS report.

    Parameters
    ----------
    mode_sources : dict
        Mapping of analysis mode → association TSV path (or pre-loaded
        record list).  Standard keys are ``"autosomal"``,
        ``"x_with_sex_covariate"``, ``"x_male_only"``,
        ``"x_female_only"``, ``"y_male_only"``.
    output_path : path-like
        Destination HTML file.
    build : str, optional
        Genome build (any alias accepted by :func:`_normalise_build`).
        Required for gene annotation.
    gene_window_kb : int
        Half-window used to collect "nearby" genes around each hit.
    top_n : int
        Number of top hits per mode.
    cache_dir : path-like, optional
        Override cache directory for UCSC downloads.
    top_hits_tsv_dir : path-like, optional
        When provided, a publication-ready top-hits TSV is written for
        each mode into this directory.

    Returns
    -------
    Path
        The HTML file that was written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load gene table if requested and possible.
    gene_table: dict | None = None
    gene_source_name: str | None = None
    if annotate_genes and build is not None:
        try:
            canon = _normalise_build(build)
            gene_table = load_gene_table(canon, cache_dir=cache_dir)
            gene_source_name = _UCSC_GENE_URLS[canon][1]
        except Exception as exc:  # network/cache failures — degrade gracefully
            logger.warning(
                "Gene annotation disabled: %s", exc,
            )
            gene_table = None

    # Build per-mode reports.  Also accumulate records for the combined
    # genome-wide summary Manhattan plot (autosomal + headline non-autosomal
    # modes: chrX sex-as-covariate, chrY males, chrM/MT sex-as-covariate).
    report_by_mode: dict[str, ModeReport] = {}
    _SUMMARY_MODES = (
        "autosomal",
        "x_with_sex_covariate",
        "y_male_only",
        "mt_with_sex_covariate",
    )
    _summary_records: list[dict] = []
    for mode, src in mode_sources.items():
        if isinstance(src, (str, Path)):
            path = Path(src)
            if not path.exists():
                logger.warning("Skipping %s — file not found: %s", mode, path)
                continue
            records = read_association_records(path)
            src_path = str(path)
        else:
            records = list(src)
            src_path = ""
        if not records:
            logger.warning("Skipping %s — no records.", mode)
            continue
        if mode in _SUMMARY_MODES:
            _summary_records.extend(records)
        report_by_mode[mode] = summarize_mode(
            mode,
            records,
            source_path=src_path,
            gene_table=gene_table,
            gene_window_kb=gene_window_kb,
            top_n=top_n,
            regional_half_window_kb=regional_half_window_kb,
            max_regional_loci=max_regional_loci,
        )

    if not report_by_mode:
        raise ValueError(
            "generate_gwas_report: no input modes produced any records."
        )

    # Build a genome-wide combined Manhattan figure that overlays the
    # autosomal scan with the headline non-autosomal scans (chrX full
    # cohort with sex as covariate, chrY males, chrM/MT full cohort with
    # sex as covariate).  Only built when more than one summary mode is
    # present; otherwise the per-mode plot already covers the full genome.
    combined_manhattan_fig: dict | None = None
    _summary_modes_present = [m for m in _SUMMARY_MODES if m in report_by_mode]
    if _summary_records and len(_summary_modes_present) >= 2:
        combined_manhattan_fig = build_manhattan_figure(
            _summary_records,
            title=(
                "Combined genome-wide summary "
                f"({', '.join(_summary_modes_present)})"
            ),
        )

    # Optional top-hits TSVs.
    if top_hits_tsv_dir is not None:
        tdir = Path(top_hits_tsv_dir)
        tdir.mkdir(parents=True, exist_ok=True)
        for mode, rep in report_by_mode.items():
            if not rep.top_hits:
                continue
            tpath = tdir / f"top_hits.{mode}.tsv"
            cols = list(rep.top_hits[0].keys())
            with open(tpath, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
                w.writeheader()
                for row in rep.top_hits:
                    w.writerow(row)
            logger.info("Wrote top hits for %s → %s", mode, tpath)

    html_doc = _render_html(
        report_by_mode=report_by_mode,
        title=title,
        build=build,
        gene_window_kb=gene_window_kb,
        gene_source=gene_source_name,
        combined_manhattan_fig=combined_manhattan_fig,
        summary_modes=_summary_modes_present,
    )
    output_path.write_text(html_doc, encoding="utf-8")
    logger.info("GWAS report written: %s (%.1f KB)", output_path,
                output_path.stat().st_size / 1024.0)
    return output_path
