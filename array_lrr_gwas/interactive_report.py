"""Interactive HTML diagnostic report for LRR correction PCs.

Generates a single self-contained HTML file with interactive Plotly charts:
  - Scree plot of singular values with Marchenko-Pastur (MP) cutoff line
  - Interactive 3D PC scatter plot (selectable PCs, overlay by sample QC metrics)
  - UMAP 2D projection

The report follows the design patterns from the NGS-PCA-Manuscript
interactive reporting script (``scripts/06_interactive_report.py``).

Usage
-----
The report is generated automatically by the ``correct`` CLI sub-command
unless ``--no-interactive-report`` is passed.  It can also be produced
programmatically::

    from array_lrr_gwas.interactive_report import generate_report

    generate_report(
        info=info,           # dict returned by correct_lrr()
        samples=samples,     # list of sample IDs
        lrr=lrr,             # original LRR matrix (markers × samples)
        output_path="report.html",
    )
"""

from __future__ import annotations

import json
import logging
import math
import textwrap
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from array_lrr_gwas.select_k import _mp_upper_edge
from array_lrr_gwas.subsetting import autosome_mask
from array_lrr_gwas.sample_sheet import read_all_raw_rows

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sample-sheet column parsing for report visualisation
# ---------------------------------------------------------------------------

def _parse_sample_sheet_columns(
    path: str | Path,
    samples: list[str],
    *,
    sample_id_col: str = "sample_id",
) -> dict[str, Any]:
    """Read all columns from a compiled sample sheet, aligned to *samples*.

    Delegates raw CSV parsing to
    :func:`~array_lrr_gwas.sample_sheet.read_all_raw_rows` (which reuses
    the shared :func:`~array_lrr_gwas.sample_sheet._resolve_column` helper),
    so there is no duplication of sheet-reading logic here.

    Parameters
    ----------
    path : path-like
        Path to a tab-separated compiled sample sheet.
    samples : list of str
        Report sample IDs (defines the output order).
    sample_id_col : str
        Column name containing sample IDs (case-insensitive lookup).

    Returns
    -------
    dict with keys:

    * ``columns`` – list of column names present in the sheet
      (excluding the sample-ID column).
    * ``numeric`` – parallel list of bool, ``True`` when the column
      contains predominantly numeric values.
    * ``data`` – dict mapping column name → per-sample list.
      Numeric columns: ``float | None`` (``None`` for missing/non-numeric
      or non-finite values).
      Categorical columns: ``str`` (empty string for missing).
    """
    other_cols, raw = read_all_raw_rows(path, sample_id_col=sample_id_col)
    if not other_cols:
        return {"columns": [], "numeric": [], "data": {}}

    # Determine numeric vs categorical per column.
    numeric_flags: list[bool] = []
    for col in other_cols:
        parseable = 0
        total = 0
        for row_data in raw.values():
            val = row_data.get(col, "").strip()
            if val != "":
                total += 1
                try:
                    float(val)
                    parseable += 1
                except ValueError:
                    pass
        # Consider numeric if ≥ 80% of non-empty values parse as float.
        numeric_flags.append(total == 0 or parseable / total >= 0.8)

    # Build per-sample aligned data lists.
    data: dict[str, list] = {col: [] for col in other_cols}
    for sid in samples:
        row_data = raw.get(sid, {})
        for col, is_num in zip(other_cols, numeric_flags):
            raw_val = row_data.get(col, "").strip()
            if is_num:
                try:
                    parsed = float(raw_val) if raw_val != "" else None
                    # Reject non-finite floats (NaN, inf) — serialize as null
                    if parsed is not None and not math.isfinite(parsed):
                        parsed = None
                    data[col].append(parsed)
                except ValueError:
                    data[col].append(None)
            else:
                data[col].append(raw_val)

    return {
        "columns": other_cols,
        "numeric": numeric_flags,
        "data": data,
    }


# ---------------------------------------------------------------------------
# Sample-level QC metrics
# ---------------------------------------------------------------------------

def compute_sample_metrics(
    lrr: NDArray[np.floating],
    samples: list[str],
    chromosomes: NDArray | Sequence[str] | None = None,
    upstream_qc_mask: NDArray[np.bool_] | None = None,
) -> dict[str, list]:
    """Compute per-sample LRR_SD and callrate using autosomal markers only.

    Parameters
    ----------
    lrr : ndarray, shape (n_markers, n_samples)
    samples : list of str
    chromosomes : array-like of str, shape (n_markers,), optional
        Chromosome label for each marker.  When provided, only autosomal
        markers are used for LRR_SD and callrate.  Non-autosomal markers
        (X, Y, MT) carry sex-linked intensity signals that inflate or
        distort per-sample QC metrics.
    upstream_qc_mask : ndarray of bool, shape (n_markers,), optional
        Pre-computed upstream variant QC mask.  When provided, the mask is
        AND-ed with the autosomal filter so that only QC-passing autosomal
        markers contribute to the metrics.

    Returns
    -------
    dict with keys ``SAMPLE``, ``LRR_SD``, ``callrate``,
    and ``n_markers_used`` (autosomal finite marker count per sample).
    LRR_SD is computed over autosomal finite values only (NaN and inf excluded).
    LRR_SD entries are ``None`` for samples with no finite autosomal markers.
    """
    # Restrict to autosomal markers when chromosome labels are available,
    # and optionally intersect with upstream variant QC mask.
    marker_mask = np.ones(lrr.shape[0], dtype=bool)
    if chromosomes is not None:
        marker_mask &= autosome_mask(chromosomes)
    if upstream_qc_mask is not None:
        marker_mask &= upstream_qc_mask
    if not marker_mask.all():
        lrr = lrr[marker_mask]
    n_markers = lrr.shape[0]
    # Use np.isfinite to exclude both NaN and inf values.
    # np.nanstd returns NaN when a column contains inf (inf - mean = NaN),
    # even if most values are finite.  Replacing non-finite values with NaN
    # first ensures nanstd only operates on valid LRR measurements.
    finite_mask = np.isfinite(lrr)
    n_valid = np.sum(finite_mask, axis=0)  # finite count per sample
    lrr_finite = np.where(finite_mask, lrr, np.nan)
    raw_sd = np.nanstd(lrr_finite, axis=0)
    lrr_sd: list[float | None] = [
        None if np.isnan(v) else float(v) for v in raw_sd
    ]
    callrate = (n_valid / n_markers).tolist()
    return {
        "SAMPLE": list(samples),
        "LRR_SD": lrr_sd,
        "callrate": callrate,
        "n_markers_used": n_valid.tolist(),
    }


def write_sample_metrics_tsv(
    metrics: dict[str, list],
    path: str | Path,
    post_metrics: dict[str, list] | None = None,
) -> Path:
    """Persist sample QC metrics to a TSV file.

    Parameters
    ----------
    metrics : dict from :func:`compute_sample_metrics`
        Pre-correction sample metrics.
    path : output file path
    post_metrics : dict from :func:`compute_sample_metrics`, optional
        Post-correction sample metrics.  When provided, ``LRR_SD_post``
        and ``callrate_post`` columns are appended for direct comparison.

    Returns
    -------
    Path to the written file.
    """
    path = Path(path)
    n_markers_used = metrics.get("n_markers_used")
    has_n_markers = n_markers_used is not None
    header = "SAMPLE\tLRR_SD\tcallrate"
    if has_n_markers:
        header += "\tn_markers_used"
    if post_metrics is not None:
        header += "\tLRR_SD_post\tcallrate_post"
    with path.open("w", encoding="utf-8") as fh:
        fh.write(header + "\n")
        for i, sample in enumerate(metrics["SAMPLE"]):
            lrr_sd_val = metrics["LRR_SD"][i]
            lrr_sd_str = "nan" if lrr_sd_val is None else f"{lrr_sd_val:.6g}"
            line = f"{sample}\t{lrr_sd_str}\t{metrics['callrate'][i]:.6g}"
            if has_n_markers:
                line += f"\t{n_markers_used[i]}"
            if post_metrics is not None:
                post_sd = post_metrics["LRR_SD"][i]
                post_sd_str = "nan" if post_sd is None else f"{post_sd:.6g}"
                line += f"\t{post_sd_str}\t{post_metrics['callrate'][i]:.6g}"
            fh.write(line + "\n")
    return path


# ---------------------------------------------------------------------------
# UMAP projection
# ---------------------------------------------------------------------------

def compute_umap(
    sample_scores: NDArray[np.floating],
    singular_values: NDArray[np.floating],
    k_mp: int,
) -> tuple[list[float], list[float]]:
    """Compute a 2-D UMAP embedding from PC scores.

    Uses ``max(3, k_mp)`` input dimensions, following NGS-PCA-Manuscript
    guidelines.

    Parameters
    ----------
    sample_scores : ndarray, shape (k, n_samples)
        Right singular vectors scaled by singular values (PC scores).
    singular_values : ndarray, shape (k,)
    k_mp : int
        Number of significant PCs from the MP cutoff.

    Returns
    -------
    (umap1, umap2) : tuple of lists
    """
    try:
        import umap as umap_module  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "umap-learn is required for the interactive report.  "
            "Install it with: pip install umap-learn"
        ) from exc

    n_dims = max(3, k_mp)
    n_dims = min(n_dims, sample_scores.shape[0])
    # PC scores: scale by singular values → (n_samples, n_dims)
    pc_scores = (singular_values[:n_dims, np.newaxis] * sample_scores[:n_dims]).T
    n_samples = pc_scores.shape[0]
    n_neighbors = min(30, max(2, n_samples - 1))
    reducer = umap_module.UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=n_neighbors,
        min_dist=0.3,
    )
    emb = reducer.fit_transform(pc_scores)
    return emb[:, 0].tolist(), emb[:, 1].tolist()


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

def _scree_data(
    singular_values: NDArray[np.floating],
    n_markers_used: int,
    n_hq_samples: int,
    k_mp: int,
) -> dict[str, Any]:
    """Prepare scree-plot data for embedding in the HTML template."""
    s2 = singular_values ** 2
    eigenvalues = s2 / n_hq_samples
    total = s2.sum()
    prop_var = (s2 / total).tolist()
    cum_var = np.cumsum(s2 / total).tolist()

    # Compute MP upper edge for the threshold line
    noise_eigenvalue = float(np.median(s2)) / n_hq_samples
    mp_threshold = _mp_upper_edge(n_markers_used, n_hq_samples, noise_eigenvalue)

    return {
        "eigenvalues": eigenvalues.tolist(),
        "prop_var": prop_var,
        "cum_var": cum_var,
        "mp_threshold": mp_threshold,
        "k_mp": k_mp,
        "n_pcs": len(singular_values),
    }


def _pc_scatter_data(
    sample_scores: NDArray[np.floating],
    singular_values: NDArray[np.floating],
    samples: list[str],
    metrics: dict[str, list],
    hq_mask: NDArray[np.bool_],
    k_mp: int,
) -> dict[str, Any]:
    """Prepare PC scatter data for the interactive 3D plot."""
    k = sample_scores.shape[0]
    pc_scores = singular_values[:, np.newaxis] * sample_scores
    pc_data: dict[str, list] = {}
    for i in range(k):
        pc_data[f"PC{i + 1}"] = pc_scores[i].tolist()
    return {
        "samples": list(samples),
        "pcs": pc_data,
        "LRR_SD": metrics["LRR_SD"],
        "callrate": metrics["callrate"],
        "hq": hq_mask.tolist(),
        "k_mp": k_mp,
        "n_pcs": k,
    }


def _lrr_comparison_data(
    pre_metrics: dict[str, list],
    post_metrics: dict[str, list],
    hq_mask: NDArray[np.bool_],
) -> dict[str, Any]:
    """Prepare pre- vs post-correction LRR comparison data for embedding."""
    return {
        "samples": pre_metrics["SAMPLE"],
        "LRR_SD_pre": pre_metrics["LRR_SD"],
        "LRR_SD_post": post_metrics["LRR_SD"],
        "callrate_pre": pre_metrics["callrate"],
        "callrate_post": post_metrics["callrate"],
        "hq": hq_mask.tolist(),
    }


def _build_html(
    scree: dict,
    scatter: dict,
    umap_data: dict | None,
    sheet_data: dict | None = None,
    lrr_comparison: dict | None = None,
    title: str = "LRR Correction Diagnostic Report",
) -> str:
    """Build a single self-contained HTML string with Plotly charts."""
    report_data = {
        "scree": scree,
        "scatter": scatter,
        "umap": umap_data,
        "sheet": sheet_data,
        "lrr_comparison": lrr_comparison,
    }
    data_json = json.dumps(report_data, allow_nan=False, default=_json_default)

    html = textwrap.dedent("""\
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>""" + title + """</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                     "Helvetica Neue", Arial, sans-serif;
        background: #f7f8fa; color: #222; line-height: 1.6;
    }
    .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
    h1 { font-size: 1.8rem; margin-bottom: 8px; color: #1a1a2e; }
    h2 { font-size: 1.3rem; margin: 24px 0 8px; color: #16213e; border-bottom: 2px solid #0f3460; padding-bottom: 4px; }
    .info { background: #e8f4f8; border-left: 4px solid #0f3460; padding: 12px 16px; margin: 12px 0; border-radius: 4px; font-size: 0.95rem; }
    .controls { background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 12px 16px; margin: 8px 0; display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }
    .controls label { font-weight: 600; font-size: 0.9rem; }
    .controls select { padding: 4px 8px; border-radius: 4px; border: 1px solid #ccc; font-size: 0.9rem; }
    .plot-container { background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 8px; margin: 8px 0; }
    footer { margin-top: 32px; padding: 16px 0; border-top: 1px solid #ddd; font-size: 0.85rem; color: #666; text-align: center; }
    </style>
    </head>
    <body>
    <div class="container">
    <h1>""" + title + """</h1>
    <div class="info">
        Interactive diagnostic report for principal components used in LRR
        batch-effect correction. Hover over points for sample details.
        Use the controls to change axes and colour overlays.
    </div>

    <!-- Scree Plot -->
    <h2>Scree Plot &mdash; Singular Values</h2>
    <div class="info">
        Eigenvalues from the SVD of the centred LRR matrix.
        The dashed red line marks the Marchenko&ndash;Pastur (MP) upper-edge
        threshold; PCs to the left are considered significant.
    </div>
    <div id="scree-plot" class="plot-container"></div>

    <!-- 3D PC Scatter -->
    <h2>Interactive PC Scatter (3-D)</h2>
    <div class="controls">
        <label>X: <select id="pc-x"></select></label>
        <label>Y: <select id="pc-y"></select></label>
        <label>Z: <select id="pc-z"></select></label>
        <label>Colour: <select id="pc-color">
            <option value="hq">HQ / LQ</option>
            <option value="LRR_SD">LRR_SD</option>
            <option value="callrate">Call Rate</option>
        </select></label>
    </div>
    <div id="pc-scatter" class="plot-container"></div>

    <!-- UMAP -->
    <h2>UMAP Projection</h2>
    <div class="info">
        2-D UMAP computed from the first <span id="umap-ndims"></span> PCs
        (max of 3 and MP-selected k).
    </div>
    <div class="controls">
        <label>Colour: <select id="umap-color">
            <option value="hq">HQ / LQ</option>
            <option value="LRR_SD">LRR_SD</option>
            <option value="callrate">Call Rate</option>
        </select></label>
    </div>
    <div id="umap-plot" class="plot-container"></div>

    <!-- LRR Comparison (only rendered when post-correction data is available) -->
    <div id="lrr-comparison-section" style="display:none;">
    <h2>Pre vs Post-Correction LRR_SD Comparison</h2>
    <div class="info">
        Scatter plot comparing pre-correction and post-correction LRR standard
        deviation for each sample.  Points on the unity line (y&nbsp;=&nbsp;x,
        dashed grey) indicate no change; points below the line indicate
        reduced noise after PC correction.  Colour: blue&nbsp;=&nbsp;HQ,
        red&nbsp;=&nbsp;LQ.
    </div>
    <div id="lrr-scatter" class="plot-container"></div>

    <h2>Bland&ndash;Altman Plot &mdash; LRR_SD Agreement</h2>
    <div class="info">
        Bland&ndash;Altman plot showing the difference
        (pre&nbsp;&minus;&nbsp;post) vs the mean of pre and post LRR_SD per
        sample.  Horizontal lines mark the mean difference (blue) and
        &pm;1.96&nbsp;SD limits of agreement (red dashed).
    </div>
    <div id="bland-altman" class="plot-container"></div>
    </div>

    <footer>
        Generated by <code>array-lrr-gwas correct</code> &mdash;
        <a href="https://github.com/jlanej/array_LRR_gwas">array_LRR_gwas</a>
    </footer>
    </div>

    <script>
    // ── Embedded data ────────────────────────────────────────────────
    const DATA = """ + data_json + """;

    // ── Sample-sheet colour options ──────────────────────────────────
    (function() {
        if (!DATA.sheet || !DATA.sheet.columns || DATA.sheet.columns.length === 0) return;
        ["pc-color", "umap-color"].forEach(function(id) {
            const sel = document.getElementById(id);
            if (!sel) return;
            DATA.sheet.columns.forEach(function(col) {
                const opt = document.createElement("option");
                opt.value = "sheet:" + col;
                opt.textContent = col;
                sel.appendChild(opt);
            });
        });
    })();

    // ── Scree plot ───────────────────────────────────────────────────
    (function() {
        const s = DATA.scree;
        const xs = Array.from({length: s.n_pcs}, (_, i) => i + 1);
        const eigenTrace = {
            x: xs, y: s.eigenvalues, type: "bar",
            marker: {color: xs.map(i => i <= s.k_mp ? "#0f3460" : "#b0bec5")},
            name: "Eigenvalue",
        };
        const cumTrace = {
            x: xs, y: s.cum_var, type: "scatter", mode: "lines+markers",
            yaxis: "y2", name: "Cumulative variance",
            line: {color: "#e94560", width: 2},
            marker: {size: 5},
        };
        const mpLine = {
            type: "line", x0: s.k_mp + 0.5, x1: s.k_mp + 0.5,
            y0: 0, y1: Math.max(...s.eigenvalues) * 1.05,
            line: {color: "red", width: 2, dash: "dash"},
        };
        const mpAnnotation = {
            x: s.k_mp + 0.5, y: Math.max(...s.eigenvalues) * 1.02,
            xanchor: "left", text: "  MP cutoff (k=" + s.k_mp + ")",
            showarrow: false, font: {color: "red", size: 12},
        };
        Plotly.newPlot("scree-plot", [eigenTrace, cumTrace], {
            xaxis: {title: "PC", dtick: 1},
            yaxis: {title: "Eigenvalue"},
            yaxis2: {title: "Cumulative variance explained", overlaying: "y", side: "right", range: [0, 1.05]},
            shapes: [mpLine],
            annotations: [mpAnnotation],
            margin: {t: 30, b: 50},
            legend: {x: 0.7, y: 0.95},
            height: 450,
        }, {responsive: true});
    })();

    // ── PC scatter helpers ───────────────────────────────────────────
    function _sheetColValues(col) {
        if (!DATA.sheet) return null;
        return DATA.sheet.data[col] || null;
    }
    function _sheetColIsNumeric(col) {
        if (!DATA.sheet) return false;
        const idx = DATA.sheet.columns.indexOf(col);
        return idx >= 0 && DATA.sheet.numeric[idx];
    }
    // Map categorical string values to discrete colours.
    // Palette cycles for columns with more than 10 unique values.
    // Missing values (null or empty string) are shown in grey (#cccccc).
    function _catColors(vals) {
        const palette = [
            "#0f3460","#e94560","#f5a623","#7ed321","#9b59b6",
            "#1abc9c","#e67e22","#3498db","#e74c3c","#2ecc71",
        ];
        const unique = [...new Set(vals.filter(v => v !== null && v !== ""))];
        unique.sort();
        const map = {};
        unique.forEach((v, i) => { map[v] = palette[i % palette.length]; });
        return vals.map(v => (v === null || v === "") ? "#cccccc" : (map[v] || "#cccccc"));
    }
    function buildColorArray(key) {
        if (key === "hq") {
            return DATA.scatter.hq.map(v => v ? "#0f3460" : "#e94560");
        }
        if (key === "LRR_SD") return DATA.scatter.LRR_SD;
        if (key === "callrate") return DATA.scatter.callrate;
        if (key.startsWith("sheet:")) {
            const col = key.slice(6);
            const vals = _sheetColValues(col);
            if (!vals) return DATA.scatter.samples.map(() => "#aaaaaa");
            if (_sheetColIsNumeric(col)) return vals;
            return _catColors(vals);
        }
        return DATA.scatter.samples.map(() => "#aaaaaa");
    }
    function colorScale(key) {
        if (key === "hq") return null;
        if (key.startsWith("sheet:")) {
            const col = key.slice(6);
            return _sheetColIsNumeric(col) ? "Viridis" : null;
        }
        return "Viridis";
    }
    function buildHoverText() {
        // Build base hover fields from scatter data.
        const baseLines = DATA.scatter.samples.map((s, i) => {
            const hq = DATA.scatter.hq[i] ? "HQ" : "LQ";
            const lrrSd = DATA.scatter.LRR_SD[i];
            const lrrSdStr = lrrSd === null ? "N/A" : lrrSd.toFixed(4);
            return s + "<br>LRR_SD=" + lrrSdStr +
                   "<br>callrate=" + DATA.scatter.callrate[i].toFixed(4) +
                   "<br>" + hq;
        });
        // Append sample-sheet columns when available.
        if (DATA.sheet && DATA.sheet.columns && DATA.sheet.columns.length > 0) {
            return baseLines.map((line, i) => {
                const extra = DATA.sheet.columns.map(col => {
                    const val = DATA.sheet.data[col][i];
                    const valStr = (val === null || val === undefined) ? "N/A" : String(val);
                    return col + "=" + valStr;
                }).join("<br>");
                return line + "<br>" + extra;
            });
        }
        return baseLines;
    }

    // ── Populate PC selectors ────────────────────────────────────────
    const nPCs = DATA.scatter.n_pcs;
    const maxSelectable = Math.max(nPCs, DATA.scatter.k_mp);
    ["pc-x", "pc-y", "pc-z"].forEach((id, idx) => {
        const sel = document.getElementById(id);
        for (let i = 1; i <= nPCs; i++) {
            const opt = document.createElement("option");
            opt.value = "PC" + i;
            opt.textContent = "PC" + i;
            if (i === idx + 1) opt.selected = true;
            sel.appendChild(opt);
        }
    });

    function render3D() {
        const xKey = document.getElementById("pc-x").value;
        const yKey = document.getElementById("pc-y").value;
        const zKey = document.getElementById("pc-z").value;
        const cKey = document.getElementById("pc-color").value;
        const colors = buildColorArray(cKey);
        const hover = buildHoverText();
        const cs = colorScale(cKey);
        const trace = {
            x: DATA.scatter.pcs[xKey],
            y: DATA.scatter.pcs[yKey],
            z: DATA.scatter.pcs[zKey],
            mode: "markers",
            type: "scatter3d",
            marker: {
                size: 4,
                color: colors,
                colorscale: cs,
                showscale: cs !== null,
                colorbar: cs !== null ? {title: cKey} : undefined,
            },
            text: hover,
            hoverinfo: "text",
        };
        const layout = {
            scene: {
                xaxis: {title: xKey},
                yaxis: {title: yKey},
                zaxis: {title: zKey},
            },
            margin: {t: 10, b: 10, l: 10, r: 10},
            height: 600,
        };
        Plotly.newPlot("pc-scatter", [trace], layout, {responsive: true});
    }
    render3D();
    ["pc-x", "pc-y", "pc-z", "pc-color"].forEach(id =>
        document.getElementById(id).addEventListener("change", render3D));

    // ── UMAP plot ────────────────────────────────────────────────────
    (function() {
        if (!DATA.umap) {
            document.getElementById("umap-plot").innerHTML =
                "<p style='padding:20px;color:#888;'>UMAP not available (install umap-learn).</p>";
            return;
        }
        document.getElementById("umap-ndims").textContent = DATA.umap.n_dims;

        function renderUMAP() {
            const cKey = document.getElementById("umap-color").value;
            const colors = buildColorArray(cKey);
            const hover = buildHoverText();
            const cs = colorScale(cKey);
            const trace = {
                x: DATA.umap.umap1,
                y: DATA.umap.umap2,
                mode: "markers",
                type: "scatter",
                marker: {
                    size: 6,
                    color: colors,
                    colorscale: cs,
                    showscale: cs !== null,
                    colorbar: cs !== null ? {title: cKey} : undefined,
                },
                text: hover,
                hoverinfo: "text",
            };
            const layout = {
                xaxis: {title: "UMAP1"},
                yaxis: {title: "UMAP2"},
                margin: {t: 20, b: 50},
                height: 500,
            };
            Plotly.newPlot("umap-plot", [trace], layout, {responsive: true});
        }
        renderUMAP();
        document.getElementById("umap-color").addEventListener("change", renderUMAP);
    })();

    // ── LRR Pre vs Post Comparison plots ────────────────────────────
    (function() {
        if (!DATA.lrr_comparison) return;
        document.getElementById("lrr-comparison-section").style.display = "";
        const comp = DATA.lrr_comparison;
        const pre = comp.LRR_SD_pre;
        const post = comp.LRR_SD_post;
        const samples = comp.samples;
        const hq = comp.hq;

        // Keep only samples with finite pre and post values.
        const vi = [];
        for (let i = 0; i < pre.length; i++) {
            if (pre[i] !== null && post[i] !== null) vi.push(i);
        }
        if (vi.length === 0) return;
        const vPre = vi.map(i => pre[i]);
        const vPost = vi.map(i => post[i]);
        const vSamples = vi.map(i => samples[i]);
        const vHq = vi.map(i => hq[i]);

        // ── Scatter: pre vs post LRR_SD ──
        const allV = vPre.concat(vPost);
        const minV = Math.min(...allV);
        const maxV = Math.max(...allV);
        const pad = (maxV - minV) * 0.05 || 0.01;
        Plotly.newPlot("lrr-scatter", [{
            x: vPre, y: vPost,
            mode: "markers", type: "scatter",
            marker: {size: 7, color: vHq.map(v => v ? "#0f3460" : "#e94560")},
            text: vSamples.map((s, i) =>
                s + "<br>Pre: " + vPre[i].toFixed(4) + "<br>Post: " + vPost[i].toFixed(4)),
            hoverinfo: "text",
        }], {
            xaxis: {title: "LRR_SD (pre-correction)", range: [minV - pad, maxV + pad]},
            yaxis: {title: "LRR_SD (post-correction)", range: [minV - pad, maxV + pad]},
            shapes: [{type: "line",
                      x0: minV - pad, y0: minV - pad,
                      x1: maxV + pad, y1: maxV + pad,
                      line: {color: "#999", width: 1.5, dash: "dash"}}],
            margin: {t: 20, b: 50, l: 60, r: 20}, height: 500,
        }, {responsive: true});

        // ── Bland-Altman: mean vs difference ──
        const means = vPre.map((v, i) => (v + vPost[i]) / 2);
        const diffs = vPre.map((v, i) => v - vPost[i]);
        const meanDiff = diffs.reduce((a, b) => a + b, 0) / diffs.length;
        const sdDiff = Math.sqrt(
            diffs.map(d => (d - meanDiff) ** 2).reduce((a, b) => a + b, 0) / diffs.length);
        const upper = meanDiff + 1.96 * sdDiff;
        const lower = meanDiff - 1.96 * sdDiff;
        const minM = Math.min(...means);
        const maxM = Math.max(...means);
        const mPad = (maxM - minM) * 0.05 || 0.01;
        Plotly.newPlot("bland-altman", [{
            x: means, y: diffs,
            mode: "markers", type: "scatter",
            marker: {size: 7, color: vHq.map(v => v ? "#0f3460" : "#e94560")},
            text: vSamples.map((s, i) =>
                s + "<br>Mean: " + means[i].toFixed(4) + "<br>Diff: " + diffs[i].toFixed(4)),
            hoverinfo: "text",
        }], {
            xaxis: {title: "Mean LRR_SD  (pre + post) / 2"},
            yaxis: {title: "Difference  (pre \u2212 post)"},
            shapes: [
                {type:"line", x0:minM-mPad, x1:maxM+mPad, y0:meanDiff, y1:meanDiff,
                 line:{color:"#0f3460", width:2}},
                {type:"line", x0:minM-mPad, x1:maxM+mPad, y0:upper, y1:upper,
                 line:{color:"#e94560", width:1.5, dash:"dash"}},
                {type:"line", x0:minM-mPad, x1:maxM+mPad, y0:lower, y1:lower,
                 line:{color:"#e94560", width:1.5, dash:"dash"}},
            ],
            annotations: [
                {x:maxM+mPad, y:meanDiff, text:"Mean: "+meanDiff.toFixed(4),
                 xanchor:"right", showarrow:false, font:{size:11,color:"#0f3460"}},
                {x:maxM+mPad, y:upper, text:"+1.96 SD",
                 xanchor:"right", showarrow:false, font:{size:11,color:"#e94560"}},
                {x:maxM+mPad, y:lower, text:"\u22121.96 SD",
                 xanchor:"right", showarrow:false, font:{size:11,color:"#e94560"}},
            ],
            margin: {t: 20, b: 50, l: 60, r: 80}, height: 500,
        }, {responsive: true});
    })();
    </script>
    </body>
    </html>
    """)
    return html


def _json_default(obj: Any) -> Any:
    """JSON serialiser fallback for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _merge_sheet_data(primary: dict, secondary: dict) -> dict:
    """Merge two sheet-data dicts, with *primary* taking precedence on name collisions.

    Parameters
    ----------
    primary, secondary : dict
        Dicts as returned by :func:`_parse_sample_sheet_columns`, each with
        keys ``columns`` (list of str), ``numeric`` (list of bool), and
        ``data`` (dict column→list).

    Returns
    -------
    dict
        Merged sheet-data dict.  If a column name exists in both sheets the
        secondary copy is suffixed with ``_illumina`` to avoid shadowing.
    """
    primary_cols: set[str] = set(primary.get("columns", []))
    merged_columns: list[str] = list(primary.get("columns", []))
    merged_numeric: list[bool] = list(primary.get("numeric", []))
    merged_data: dict = dict(primary.get("data", {}))

    for col, is_num in zip(secondary.get("columns", []), secondary.get("numeric", [])):
        dest = col if col not in primary_cols else f"{col}_illumina"
        merged_columns.append(dest)
        merged_numeric.append(is_num)
        merged_data[dest] = secondary["data"][col]

    return {"columns": merged_columns, "numeric": merged_numeric, "data": merged_data}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(
    info: dict,
    samples: list[str],
    lrr: NDArray[np.floating],
    output_path: str | Path,
    *,
    corrected_lrr: NDArray[np.floating] | None = None,
    post_metrics: dict[str, list] | None = None,
    chromosomes: NDArray | Sequence[str] | None = None,
    upstream_qc_mask: NDArray[np.bool_] | None = None,
    metrics_tsv_path: str | Path | None = None,
    sample_sheet_path: str | Path | None = None,
    illumina_sample_sheet_path: str | Path | None = None,
    illumina_sample_id_col: str = "Sample_Group",
    skip_umap: bool = False,
) -> Path:
    """Generate an interactive HTML diagnostic report.

    Parameters
    ----------
    info : dict
        Metadata dict returned by :func:`~array_lrr_gwas.correction.correct_lrr`.
        Must contain ``singular_values``, ``sample_scores``, ``hq_sample_mask``,
        ``k``, ``n_hq_samples``, and ``n_markers_used``.
    samples : list of str
        Sample ID strings.
    lrr : ndarray, shape (n_markers, n_samples)
        Original (uncorrected) LRR matrix.
    output_path : path-like
        Where to write the HTML file.
    corrected_lrr : ndarray, shape (n_markers, n_samples), optional
        Post-correction LRR matrix.  When provided, post-correction LRR_SD
        and callrate are computed on the same marker set as pre-correction
        and included in the metrics TSV and interactive comparison plots
        (scatter and Bland–Altman).
    post_metrics : dict, optional
        Pre-computed post-correction sample metrics (with keys ``SAMPLE``,
        ``LRR_SD``, ``callrate``, ``n_markers_used``), as returned by
        :func:`compute_sample_metrics` or accumulated during streaming
        correction.  When provided, these are used directly instead of
        computing from *corrected_lrr*.  Takes precedence over
        *corrected_lrr* if both are given.
    chromosomes : array-like of str, shape (n_markers,), optional
        Chromosome label for each marker.  When provided, only autosomal
        markers are used for LRR_SD and callrate in the sample metrics.
    upstream_qc_mask : ndarray of bool, shape (n_markers,), optional
        Pre-computed upstream variant QC mask.  When provided, only
        QC-passing autosomal markers contribute to sample metrics,
        ensuring consistency between pre- and post-correction comparisons.
    metrics_tsv_path : path-like or None
        If provided, sample metrics (LRR_SD, callrate) are also saved to this TSV.
    sample_sheet_path : path-like or None
        Optional path to a compiled sample sheet TSV.  When provided, all
        columns are parsed and embedded in the report for use as colour
        overlays in the PC scatter and UMAP plots.
    illumina_sample_sheet_path : path-like or None
        Optional path to an Illumina-format SampleSheet.csv (comma-separated
        with ``[Header]``/``[Data]`` section markers).  Columns from the
        ``[Data]`` section are embedded alongside any columns from
        *sample_sheet_path*.  Both sheets may be supplied at the same time.
        On column-name collision the compiled sheet takes precedence and the
        Illumina column is suffixed with ``_illumina``.
    illumina_sample_id_col : str
        Column in the Illumina SampleSheet.csv ``[Data]`` section whose values
        match the sample IDs in *samples*.  Defaults to ``"Sample_Group"``
        because Illumina sheets typically store the short population/sample
        identifier (e.g. ``NA19152``) in that column while ``Sample_ID``
        holds a longer Illumina-internal barcode string.  If the column is not
        found the first column of the ``[Data]`` section is used as a fallback.
    skip_umap : bool
        If ``True``, skip the UMAP computation (useful when umap-learn is
        not installed or the dataset is very small).

    Returns
    -------
    Path to the written HTML file.
    """
    output_path = Path(output_path)
    singular_values = np.asarray(info["singular_values"])
    sample_scores = np.asarray(info["sample_scores"])
    hq_mask = np.asarray(info["hq_sample_mask"])
    k_mp = int(info["k"])
    n_hq = int(info["n_hq_samples"])
    n_markers = int(info["n_markers_used"])

    # 1. Compute pre-correction sample metrics
    metrics = compute_sample_metrics(
        lrr, samples, chromosomes=chromosomes,
        upstream_qc_mask=upstream_qc_mask,
    )

    # 1b. Compute or use pre-computed post-correction metrics
    _post: dict[str, list] | None = post_metrics
    lrr_comparison: dict | None = None
    if _post is None and corrected_lrr is not None:
        _post = compute_sample_metrics(
            corrected_lrr, samples, chromosomes=chromosomes,
            upstream_qc_mask=upstream_qc_mask,
        )
        logger.info(
            "Computed post-correction LRR metrics for %d samples",
            len(samples),
        )
    if _post is not None:
        lrr_comparison = _lrr_comparison_data(metrics, _post, hq_mask)
        if post_metrics is not None:
            logger.info(
                "Using pre-computed post-correction metrics for %d samples",
                len(samples),
            )

    if metrics_tsv_path is not None:
        write_sample_metrics_tsv(metrics, metrics_tsv_path, post_metrics=_post)
        logger.info("Wrote sample metrics: %s", metrics_tsv_path)

    # 2. Scree data
    scree = _scree_data(singular_values, n_markers, n_hq, k_mp)

    # 3. PC scatter data
    scatter = _pc_scatter_data(
        sample_scores, singular_values, samples, metrics, hq_mask, k_mp
    )

    # 4. UMAP
    umap_data = None
    if not skip_umap:
        try:
            u1, u2 = compute_umap(sample_scores, singular_values, k_mp)
            n_dims = min(max(3, k_mp), sample_scores.shape[0])
            umap_data = {"umap1": u1, "umap2": u2, "n_dims": n_dims}
        except ImportError:
            logger.warning(
                "umap-learn not installed; skipping UMAP in report. "
                "Install with: pip install umap-learn"
            )
        except Exception:
            logger.warning("UMAP computation failed; skipping.", exc_info=True)

    # 5. Sample-sheet columns (compiled TSV and/or Illumina CSV, both optional)
    sheet_data: dict | None = None

    def _try_parse(path: "str | Path", label: str, *, sample_id_col: str = "sample_id") -> "dict | None":
        try:
            result = _parse_sample_sheet_columns(path, samples, sample_id_col=sample_id_col)
            logger.info(
                "Loaded %d %s columns for report visualisation",
                len(result.get("columns", [])),
                label,
            )
            return result
        except Exception:
            logger.warning(
                "Failed to parse %s %s for report; skipping.",
                label,
                path,
                exc_info=True,
            )
            return None

    compiled = _try_parse(sample_sheet_path, "compiled sample sheet") if sample_sheet_path is not None else None
    illumina = _try_parse(illumina_sample_sheet_path, "Illumina sample sheet", sample_id_col=illumina_sample_id_col) if illumina_sample_sheet_path is not None else None

    if compiled is not None and illumina is not None:
        sheet_data = _merge_sheet_data(compiled, illumina)
    elif compiled is not None:
        sheet_data = compiled
    elif illumina is not None:
        sheet_data = illumina

    # 6. Build HTML
    html = _build_html(scree, scatter, umap_data, sheet_data,
                       lrr_comparison=lrr_comparison)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Interactive report written to %s", output_path)
    return output_path
