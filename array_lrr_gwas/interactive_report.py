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
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from array_lrr_gwas.select_k import _mp_upper_edge

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sample-level QC metrics
# ---------------------------------------------------------------------------

def compute_sample_metrics(
    lrr: NDArray[np.floating],
    samples: list[str],
) -> dict[str, list]:
    """Compute per-sample LRR_SD and callrate.

    Parameters
    ----------
    lrr : ndarray, shape (n_markers, n_samples)
    samples : list of str

    Returns
    -------
    dict with keys ``SAMPLE``, ``LRR_SD``, ``callrate``,
    and ``n_markers_used`` (finite, non-NaN marker count per sample).
    LRR_SD is computed over finite values only (NaN and inf excluded).
    LRR_SD entries are ``None`` for samples with no finite markers.
    """
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
) -> Path:
    """Persist sample QC metrics to a TSV file.

    Parameters
    ----------
    metrics : dict from :func:`compute_sample_metrics`
    path : output file path

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
    with path.open("w", encoding="utf-8") as fh:
        fh.write(header + "\n")
        for i, sample in enumerate(metrics["SAMPLE"]):
            lrr_sd_val = metrics["LRR_SD"][i]
            lrr_sd_str = "nan" if lrr_sd_val is None else f"{lrr_sd_val:.6g}"
            line = f"{sample}\t{lrr_sd_str}\t{metrics['callrate'][i]:.6g}"
            if has_n_markers:
                line += f"\t{n_markers_used[i]}"
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


def _build_html(
    scree: dict,
    scatter: dict,
    umap_data: dict | None,
    title: str = "LRR Correction Diagnostic Report",
) -> str:
    """Build a single self-contained HTML string with Plotly charts."""
    report_data = {
        "scree": scree,
        "scatter": scatter,
        "umap": umap_data,
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

    <footer>
        Generated by <code>array-lrr-gwas correct</code> &mdash;
        <a href="https://github.com/jlanej/array_LRR_gwas">array_LRR_gwas</a>
    </footer>
    </div>

    <script>
    // ── Embedded data ────────────────────────────────────────────────
    const DATA = """ + data_json + """;

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
        }, {responsive: true});
    })();

    // ── PC scatter helpers ───────────────────────────────────────────
    function buildColorArray(key) {
        if (key === "hq") {
            return DATA.scatter.hq.map(v => v ? "#0f3460" : "#e94560");
        }
        const vals = key === "LRR_SD" ? DATA.scatter.LRR_SD : DATA.scatter.callrate;
        return vals;
    }
    function colorScale(key) {
        if (key === "hq") return null;
        return "Viridis";
    }
    function buildHoverText() {
        return DATA.scatter.samples.map((s, i) => {
            const hq = DATA.scatter.hq[i] ? "HQ" : "LQ";
            const lrrSd = DATA.scatter.LRR_SD[i];
            const lrrSdStr = lrrSd === null ? "N/A" : lrrSd.toFixed(4);
            return s + "<br>LRR_SD=" + lrrSdStr +
                   "<br>callrate=" + DATA.scatter.callrate[i].toFixed(4) +
                   "<br>" + hq;
        });
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
        const trace = {
            x: DATA.scatter.pcs[xKey],
            y: DATA.scatter.pcs[yKey],
            z: DATA.scatter.pcs[zKey],
            mode: "markers",
            type: "scatter3d",
            marker: {
                size: 4,
                color: colors,
                colorscale: colorScale(cKey),
                showscale: cKey !== "hq",
                colorbar: cKey !== "hq" ? {title: cKey} : undefined,
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
            const trace = {
                x: DATA.umap.umap1,
                y: DATA.umap.umap2,
                mode: "markers",
                type: "scatter",
                marker: {
                    size: 6,
                    color: colors,
                    colorscale: colorScale(cKey),
                    showscale: cKey !== "hq",
                    colorbar: cKey !== "hq" ? {title: cKey} : undefined,
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(
    info: dict,
    samples: list[str],
    lrr: NDArray[np.floating],
    output_path: str | Path,
    *,
    metrics_tsv_path: str | Path | None = None,
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
    metrics_tsv_path : path-like or None
        If provided, sample metrics (LRR_SD, callrate) are also saved to this TSV.
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

    # 1. Compute sample metrics
    metrics = compute_sample_metrics(lrr, samples)
    if metrics_tsv_path is not None:
        write_sample_metrics_tsv(metrics, metrics_tsv_path)
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

    # 5. Build HTML
    html = _build_html(scree, scatter, umap_data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Interactive report written to %s", output_path)
    return output_path
