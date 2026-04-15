"""BCF / VCF I/O utilities for batch-corrected LRR.

Reads LRR values from a BCF/VCF ``FORMAT/LRR`` field, applies batch-effect
correction via :func:`array_lrr_gwas.correction.correct_lrr`, and writes a
new BCF/VCF with the corrected values and a descriptive header entry.

Provides both full-load and memory-efficient streaming modes:

- :func:`read_lrr` loads the entire LRR matrix at once.
- :func:`read_lrr_selected` loads only variants whose ID appears in a
  caller-supplied set, enabling RAM-budgeted workflows.
- :func:`stream_correct_write` streams through the input BCF/VCF applying
  QR-based PC regression one chunk at a time and writing corrected values
  directly, so the full matrix is never resident in memory.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

try:
    import pysam
except ImportError as _exc:
    raise ImportError(
        "pysam is required for BCF/VCF I/O. Install with: pip install pysam"
    ) from _exc

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def read_lrr(
    path: str | Path,
) -> tuple[NDArray[np.floating], list[str], list[dict]]:
    """Read LRR values and variant metadata from a BCF/VCF file.

    Parameters
    ----------
    path : str or Path
        Path to a BCF or VCF file (optionally compressed).

    Returns
    -------
    lrr : ndarray, shape (n_variants, n_samples)
        LRR values; missing entries are ``np.nan``.
    samples : list of str
        Sample identifiers in column order.
    variants : list of dict
        Per-variant metadata: ``chrom``, ``pos``, ``id``, ``ref``, ``alts``,
        ``qual``, ``filter``, ``intensity_only``.
    """
    path = str(path)
    vcf_in = pysam.VariantFile(path)
    samples = list(vcf_in.header.samples)
    n_samples = len(samples)

    lrr_rows: list[NDArray] = []
    variants: list[dict] = []
    for rec in vcf_in:
        row = np.full(n_samples, np.nan, dtype=np.float64)
        for i, sample_data in enumerate(rec.samples.values()):
            val = sample_data.get("LRR")
            if val is not None:
                try:
                    v = float(val)
                    if np.isfinite(v):
                        row[i] = v
                except (TypeError, ValueError):
                    pass
        lrr_rows.append(row)
        # Check for INTENSITY_ONLY INFO flag.  The flag is set by the
        # upstream pipeline for non-polymorphic probes that report
        # intensity but have no genotype cluster (no GT field).
        # When the INFO key is not defined in the header, pysam raises
        # ValueError — treat as False in that case.
        try:
            intensity_only = bool(rec.info.get("INTENSITY_ONLY", False))
        except (ValueError, KeyError):
            intensity_only = False
        variants.append(
            {
                "chrom": rec.chrom,
                "pos": rec.pos,
                "id": rec.id,
                "ref": rec.ref,
                "alts": tuple(rec.alts) if rec.alts else (),
                "qual": rec.qual,
                "filter": list(rec.filter),
                "intensity_only": intensity_only,
            }
        )
    vcf_in.close()

    if not lrr_rows:
        lrr = np.empty((0, n_samples), dtype=np.float64)
    else:
        lrr = np.vstack(lrr_rows)
    return lrr, samples, variants


def _variant_id_from_rec(rec) -> str:
    """Build a canonical variant ID string from a pysam VariantRecord."""
    vid = rec.id
    if vid is not None and vid != ".":
        return vid
    alts = tuple(rec.alts) if rec.alts else ()
    return f"{rec.chrom}:{rec.pos}:{rec.ref}:{':'.join(alts)}"


def _parse_rec_row(rec, n_samples: int, samples: list[str]):
    """Parse a single pysam record into an LRR row and variant metadata."""
    row = np.full(n_samples, np.nan, dtype=np.float64)
    for i, sample_data in enumerate(rec.samples.values()):
        val = sample_data.get("LRR")
        if val is not None:
            try:
                v = float(val)
                if np.isfinite(v):
                    row[i] = v
            except (TypeError, ValueError):
                pass
    try:
        intensity_only = bool(rec.info.get("INTENSITY_ONLY", False))
    except (ValueError, KeyError):
        intensity_only = False
    var_meta = {
        "chrom": rec.chrom,
        "pos": rec.pos,
        "id": rec.id,
        "ref": rec.ref,
        "alts": tuple(rec.alts) if rec.alts else (),
        "qual": rec.qual,
        "filter": list(rec.filter),
        "intensity_only": intensity_only,
    }
    return row, var_meta


def read_bcf_sample_ids(
    path: str | Path,
) -> list[str]:
    """Read sample identifiers from a BCF/VCF header without loading data.

    Parameters
    ----------
    path : str or Path
        Path to a BCF or VCF file.

    Returns
    -------
    samples : list of str
        Sample identifiers in column order.
    """
    vcf_in = pysam.VariantFile(str(path))
    samples = list(vcf_in.header.samples)
    vcf_in.close()
    return samples


def read_lrr_selected(
    path: str | Path,
    selected_ids: set[str],
) -> tuple[NDArray[np.floating], list[str], list[dict]]:
    """Read LRR values for a pre-selected subset of variants.

    Only variants whose canonical ID (see :func:`read_lrr`) appears in
    *selected_ids* are loaded.  All other variants are skipped, keeping
    peak memory proportional to ``len(selected_ids) × n_samples``.

    Parameters
    ----------
    path : str or Path
        Path to a BCF or VCF file (optionally compressed).
    selected_ids : set of str
        Variant IDs to include.  Variants not in this set are skipped.

    Returns
    -------
    lrr : ndarray, shape (n_selected, n_samples)
        LRR values for the selected variants; missing entries ``np.nan``.
    samples : list of str
        Sample identifiers in column order.
    variants : list of dict
        Per-variant metadata (same keys as :func:`read_lrr`).
    """
    path = str(path)
    vcf_in = pysam.VariantFile(path)
    samples = list(vcf_in.header.samples)
    n_samples = len(samples)

    lrr_rows: list[NDArray] = []
    variants: list[dict] = []
    n_total = 0
    with tqdm(
        desc="Loading selected markers",
        unit="variant",
        leave=False,
        dynamic_ncols=True,
    ) as pbar:
        for rec in vcf_in:
            n_total += 1
            vid = _variant_id_from_rec(rec)
            if vid not in selected_ids:
                pbar.update(1)
                continue
            row, var_meta = _parse_rec_row(rec, n_samples, samples)
            lrr_rows.append(row)
            variants.append(var_meta)
            pbar.update(1)
            pbar.set_postfix(
                loaded=len(lrr_rows),
                scanned=n_total,
                refresh=False,
            )
    vcf_in.close()

    logger.info(
        "Loaded %d / %d variants matching selection (skipped %d)",
        len(lrr_rows), n_total, n_total - len(lrr_rows),
    )

    if not lrr_rows:
        lrr = np.empty((0, n_samples), dtype=np.float64)
    else:
        lrr = np.vstack(lrr_rows)
    return lrr, samples, variants


def read_variant_metadata(
    path: str | Path,
) -> tuple[list[str], list[dict]]:
    """Read sample IDs and variant metadata from a BCF/VCF without loading LRR.

    This is a lightweight scan that extracts only variant-level metadata
    (chrom, pos, id, ref, alts, intensity_only flag, etc.) and the sample
    identifiers.  No ``FORMAT/LRR`` values are read, so memory usage is
    independent of matrix size.

    Parameters
    ----------
    path : str or Path
        Path to a BCF or VCF file (optionally compressed).

    Returns
    -------
    samples : list of str
        Sample identifiers in column order.
    variants : list of dict
        Per-variant metadata: ``chrom``, ``pos``, ``id``, ``ref``, ``alts``,
        ``qual``, ``filter``, ``intensity_only``.
    """
    path = str(path)
    vcf_in = pysam.VariantFile(path)
    samples = list(vcf_in.header.samples)

    variants: list[dict] = []
    for rec in vcf_in:
        try:
            intensity_only = bool(rec.info.get("INTENSITY_ONLY", False))
        except (ValueError, KeyError):
            intensity_only = False
        variants.append(
            {
                "chrom": rec.chrom,
                "pos": rec.pos,
                "id": rec.id,
                "ref": rec.ref,
                "alts": tuple(rec.alts) if rec.alts else (),
                "qual": rec.qual,
                "filter": list(rec.filter),
                "intensity_only": intensity_only,
            }
        )
    vcf_in.close()
    return samples, variants


def stream_lrr_chunks(
    path: str | Path,
    *,
    chunk_size: int = 5000,
    sample_mask: NDArray[np.bool_] | None = None,
    variant_mask: NDArray[np.bool_] | None = None,
):
    """Yield LRR data in fixed-size chunks without holding the full matrix.

    Each yielded element is a tuple ``(lrr_chunk, variants_chunk)`` where
    ``lrr_chunk`` has shape ``(<=chunk_size, n_selected_samples)`` and
    ``variants_chunk`` is the corresponding list of variant metadata dicts.

    Parameters
    ----------
    path : str or Path
        Path to a BCF or VCF file (optionally compressed).
    chunk_size : int
        Maximum number of variants per yielded chunk (default: 5000).
    sample_mask : ndarray of bool, optional
        Boolean mask selecting which samples to retain.  When ``None``
        all samples are included.
    variant_mask : ndarray of bool, optional
        Boolean mask selecting which variants to include.  Variants where
        the mask is ``False`` are skipped.  When ``None`` all variants
        are included.

    Yields
    ------
    lrr_chunk : ndarray, shape (<=chunk_size, n_selected_samples)
        LRR values for the chunk; missing entries are ``np.nan``.
    variants_chunk : list of dict
        Per-variant metadata for the chunk.
    """
    path = str(path)
    vcf_in = pysam.VariantFile(path)
    samples = list(vcf_in.header.samples)
    n_samples = len(samples)

    # Determine which sample columns to keep.
    if sample_mask is not None:
        keep_cols = np.flatnonzero(sample_mask)
        n_out = len(keep_cols)
    else:
        keep_cols = None
        n_out = n_samples

    lrr_rows: list[NDArray] = []
    var_chunk: list[dict] = []

    variant_idx = 0
    for rec in vcf_in:
        # Skip variants not in mask
        if variant_mask is not None and not variant_mask[variant_idx]:
            variant_idx += 1
            continue
        variant_idx += 1

        row = np.full(n_samples, np.nan, dtype=np.float64)
        for i, sample_data in enumerate(rec.samples.values()):
            val = sample_data.get("LRR")
            if val is not None:
                try:
                    v = float(val)
                    if np.isfinite(v):
                        row[i] = v
                except (TypeError, ValueError):
                    pass

        if keep_cols is not None:
            row = row[keep_cols]

        try:
            intensity_only = bool(rec.info.get("INTENSITY_ONLY", False))
        except (ValueError, KeyError):
            intensity_only = False

        var_meta = {
            "chrom": rec.chrom,
            "pos": rec.pos,
            "id": rec.id,
            "ref": rec.ref,
            "alts": tuple(rec.alts) if rec.alts else (),
            "qual": rec.qual,
            "filter": list(rec.filter),
            "intensity_only": intensity_only,
        }

        lrr_rows.append(row)
        var_chunk.append(var_meta)

        if len(lrr_rows) >= chunk_size:
            yield np.vstack(lrr_rows), var_chunk
            lrr_rows = []
            var_chunk = []

    vcf_in.close()

    # Yield any remaining rows
    if lrr_rows:
        yield np.vstack(lrr_rows), var_chunk


def stream_lrr_contig_chunks(
    path: str | Path,
    contigs: list[str],
    *,
    chunk_size: int = 5000,
    sample_mask: NDArray[np.bool_] | None = None,
):
    """Yield LRR data for specific contigs using pysam region fetch.

    When the BCF/VCF is indexed (CSI or TBI), this uses
    ``pysam.VariantFile.fetch(contig)`` to jump directly to the relevant
    chromosomes instead of streaming the entire file.  If the file is not
    indexed, falls back to a sequential scan skipping non-matching records.

    Parameters
    ----------
    path : str or Path
        Path to a BCF or VCF file.  For fast random access, an index file
        (``.csi`` or ``.tbi``) must be present alongside.
    contigs : list of str
        Chromosome/contig names to fetch (e.g. ``["chrX", "X"]``).
    chunk_size : int
        Maximum number of variants per yielded chunk (default: 5000).
    sample_mask : ndarray of bool, optional
        Boolean mask selecting which samples to retain.

    Yields
    ------
    lrr_chunk : ndarray, shape (<=chunk_size, n_selected_samples)
    variants_chunk : list of dict
    """
    path = str(path)
    vcf_in = pysam.VariantFile(path)
    samples_list = list(vcf_in.header.samples)
    n_samples = len(samples_list)

    if sample_mask is not None:
        keep_cols = np.flatnonzero(sample_mask)
        n_out = len(keep_cols)
    else:
        keep_cols = None
        n_out = n_samples

    # Check whether the file is indexed (has a CSI / TBI index).
    _is_indexed = vcf_in.index is not None

    lrr_rows: list[NDArray] = []
    var_chunk: list[dict] = []

    def _process_record(rec):
        row = np.full(n_samples, np.nan, dtype=np.float64)
        for i, sample_data in enumerate(rec.samples.values()):
            val = sample_data.get("LRR")
            if val is not None:
                try:
                    v = float(val)
                    if np.isfinite(v):
                        row[i] = v
                except (TypeError, ValueError):
                    pass
        if keep_cols is not None:
            row = row[keep_cols]
        try:
            intensity_only = bool(rec.info.get("INTENSITY_ONLY", False))
        except (ValueError, KeyError):
            intensity_only = False
        return row, {
            "chrom": rec.chrom,
            "pos": rec.pos,
            "id": rec.id,
            "ref": rec.ref,
            "alts": tuple(rec.alts) if rec.alts else (),
            "qual": rec.qual,
            "filter": list(rec.filter),
            "intensity_only": intensity_only,
        }

    contig_set = set(contigs)

    if _is_indexed:
        # Fast path: fetch each contig in turn without scanning the whole file.
        seen: set[str] = set()
        for contig in contigs:
            if contig in seen:
                continue
            seen.add(contig)
            try:
                for rec in vcf_in.fetch(contig=contig):
                    row, meta = _process_record(rec)
                    lrr_rows.append(row)
                    var_chunk.append(meta)
                    if len(lrr_rows) >= chunk_size:
                        yield np.vstack(lrr_rows), var_chunk
                        lrr_rows = []
                        var_chunk = []
            except (ValueError, KeyError):
                # Contig not present in this file — skip silently.
                pass
    else:
        # Slow path: sequential scan, filter by contig name.
        for rec in vcf_in:
            if rec.chrom not in contig_set:
                continue
            row, meta = _process_record(rec)
            lrr_rows.append(row)
            var_chunk.append(meta)
            if len(lrr_rows) >= chunk_size:
                yield np.vstack(lrr_rows), var_chunk
                lrr_rows = []
                var_chunk = []

    vcf_in.close()

    if lrr_rows:
        yield np.vstack(lrr_rows), var_chunk


def stream_correct_write(
    path_in: str | Path,
    path_out: str | Path,
    Vt_k: NDArray[np.floating],
    samples: Sequence[str],
    correction_info: dict,
    *,
    path_template: str | Path | None = None,
    chunk_size: int = 5000,
    min_valid_frac: float = 0.5,
    diagnostic_marker_ids: set[str] | None = None,
) -> tuple[list[dict], int, dict | None]:
    """Stream through a BCF/VCF, apply QR PC-regression, and write output.

    This function never holds the full LRR matrix in memory.  It reads
    the input file in chunks of *chunk_size* variants, applies the
    precomputed QR regression (from :func:`~array_lrr_gwas.correction.qr_precompute`),
    and writes corrected values directly to *path_out*.

    When *diagnostic_marker_ids* is provided, post-correction sample QC
    metrics (LRR_SD and callrate) are computed progressively during
    streaming for the matching markers.  This avoids a separate in-memory
    correction pass solely for diagnostic reporting.

    Parameters
    ----------
    path_in : str or Path
        Input BCF/VCF file with ``FORMAT/LRR``.
    path_out : str or Path
        Output BCF (``.bcf``) or VCF file.
    Vt_k : ndarray, shape (k, n_samples)
        PC scores (first *k* right singular vectors) for all samples.
    samples : sequence of str
        Sample identifiers in column order (must match input file).
    correction_info : dict
        Correction metadata (from :func:`~array_lrr_gwas.correction.correct_lrr`).
    path_template : str or Path or None
        Optional input file used as a header template.
    chunk_size : int
        Number of variant rows processed per QR-regression batch.
    min_valid_frac : float
        Minimum fraction of finite samples required for correction.
    diagnostic_marker_ids : set of str or None
        When provided, accumulate post-correction LRR_SD and callrate
        for these markers during streaming.  The returned *post_metrics*
        dict will contain per-sample statistics computed on this subset.

    Returns
    -------
    all_variants : list of dict
        Per-variant metadata for every variant in the file.
    n_skipped : int
        Number of markers left uncorrected (insufficient valid data).
    post_metrics : dict or None
        When *diagnostic_marker_ids* is given, a dict with keys
        ``SAMPLE``, ``LRR_SD``, ``callrate``, and ``n_markers_used``,
        matching the format of
        :func:`~array_lrr_gwas.interactive_report.compute_sample_metrics`.
        ``None`` when *diagnostic_marker_ids* is not provided.
    """
    from array_lrr_gwas.correction import qr_precompute, _correct_chunk_qr

    path_in = str(path_in)
    path_out_str = str(path_out)
    is_bcf = path_out_str.endswith(".bcf")

    # Precompute QR decomposition
    Q, R, X = qr_precompute(Vt_k)
    k = Vt_k.shape[0]
    n_samples = len(samples)

    # Build output header
    if path_template is not None:
        template = pysam.VariantFile(str(path_template))
        hdr = template.header.copy()
        template.close()
    else:
        hdr = pysam.VariantHeader()

    if "LRR" not in hdr.formats:
        hdr.add_meta(
            "FORMAT",
            items=[
                ("ID", "LRR"), ("Number", "1"),
                ("Type", "Float"), ("Description", "Log R Ratio"),
            ],
        )
    if "GT" not in hdr.formats:
        hdr.add_meta(
            "FORMAT",
            items=[
                ("ID", "GT"), ("Number", "1"),
                ("Type", "String"), ("Description", "Genotype"),
            ],
        )
    desc = _build_header_description(correction_info)
    hdr.add_meta(key=_CORRECTION_HEADER_KEY, value=desc)
    for s in samples:
        if s not in hdr.samples:
            hdr.add_sample(s)

    # Copy contig definitions from input header (avoids a full file scan)
    vcf_scan = pysam.VariantFile(path_in)
    for c in vcf_scan.header.contigs:
        if c not in hdr.contigs:
            hdr.add_meta("contig", items=[("ID", c)])
    vcf_scan.close()

    # Open input for streaming
    vcf_in = pysam.VariantFile(path_in)
    mode = "wb" if is_bcf else "w"
    vcf_out = pysam.VariantFile(path_out_str, mode, header=hdr)

    all_variants: list[dict] = []
    total_skipped = 0

    # Online accumulators for diagnostic marker metrics.
    # We track per-sample: count of finite corrected values, running sum,
    # and running sum-of-squares so that LRR_SD and callrate can be
    # computed at the end without storing the corrected values.
    _diag_n_total = 0  # total diagnostic markers seen (denominator for callrate)
    _diag_count: NDArray | None = None  # (n_samples,) finite count
    _diag_sum: NDArray | None = None    # (n_samples,) running sum
    _diag_sum_sq: NDArray | None = None  # (n_samples,) running sum of squares
    if diagnostic_marker_ids is not None:
        _diag_count = np.zeros(n_samples, dtype=np.int64)
        _diag_sum = np.zeros(n_samples, dtype=np.float64)
        _diag_sum_sq = np.zeros(n_samples, dtype=np.float64)

    # Buffer for chunk-wise processing
    chunk_rows: list[NDArray] = []
    chunk_recs: list[dict] = []

    # Cache sample list once outside the loop to avoid repeated conversion
    _samples_list = list(samples)

    logger.info(
        "Streaming PC correction: regressing %d PCs out of "
        "%d samples in chunk(s) of %d",
        k, n_samples, chunk_size,
    )

    def _flush_chunk(rows, recs):
        """Apply QR correction to buffered rows and write to output."""
        nonlocal total_skipped, _diag_n_total
        if not rows:
            return
        chunk = np.vstack(rows)
        corrected_chunk, chunk_skipped = _correct_chunk_qr(
            chunk, Q, X, min_valid_frac=min_valid_frac,
        )
        total_skipped += chunk_skipped

        # Identify which rows in this chunk are diagnostic markers
        diag_row_indices: list[int] | None = None
        if diagnostic_marker_ids is not None:
            diag_row_indices = []
            for i, var_meta in enumerate(recs):
                vid = var_meta.get("id")
                if vid is None or vid == ".":
                    alts = var_meta.get("alts") or ()
                    vid = (
                        f"{var_meta['chrom']}:{var_meta['pos']}"
                        f":{var_meta.get('ref', '')}:{':'.join(alts)}"
                    )
                if vid in diagnostic_marker_ids:
                    diag_row_indices.append(i)

        for i, var_meta in enumerate(recs):
            # pysam requires at least 2 alleles (REF + ALT).  Intensity-only
            # markers from illumina_idat_processing have no ALT allele
            # (rec.alts is None), so we must supply the VCF missing-ALT
            # placeholder "." to avoid ``ValueError: must set at least
            # 2 alleles``.
            alts = tuple(var_meta.get("alts", ()))
            if not alts:
                alts = (".",)
            out_rec = vcf_out.new_record(
                contig=var_meta["chrom"],
                start=var_meta["pos"] - 1,
                stop=var_meta["pos"],
                alleles=(var_meta["ref"],) + alts,
                id=var_meta.get("id"),
            )
            for j, sample_data in enumerate(out_rec.samples.values()):
                val = corrected_chunk[i, j]
                if np.isnan(val):
                    sample_data["LRR"] = None
                else:
                    sample_data["LRR"] = float(val)
            vcf_out.write(out_rec)

        # Accumulate post-correction stats for diagnostic markers
        if diag_row_indices:
            diag_chunk = corrected_chunk[diag_row_indices]  # (n_diag, n_samples)
            _diag_n_total += len(diag_row_indices)
            finite = np.isfinite(diag_chunk)
            safe = np.where(finite, diag_chunk, 0.0)
            _diag_count[:] += finite.sum(axis=0)
            _diag_sum[:] += safe.sum(axis=0)
            _diag_sum_sq[:] += (safe * safe).sum(axis=0)

    with tqdm(
        desc="Streaming PC regression",
        unit="marker",
        leave=False,
        dynamic_ncols=True,
    ) as pbar:
        for rec in vcf_in:
            row, var_meta = _parse_rec_row(rec, n_samples, _samples_list)
            all_variants.append(var_meta)
            chunk_rows.append(row)
            chunk_recs.append(var_meta)

            if len(chunk_rows) >= chunk_size:
                _flush_chunk(chunk_rows, chunk_recs)
                pbar.update(len(chunk_rows))
                pbar.set_postfix(
                    skipped=total_skipped,
                    refresh=False,
                )
                chunk_rows.clear()
                chunk_recs.clear()

        # Flush remaining
        if chunk_rows:
            _flush_chunk(chunk_rows, chunk_recs)
            pbar.update(len(chunk_rows))

    vcf_in.close()
    vcf_out.close()

    n_total_variants = len(all_variants)
    if total_skipped > 0:
        logger.info(
            "Streaming QR regression: %d / %d markers skipped "
            "(insufficient valid data)",
            total_skipped, n_total_variants,
        )
    else:
        logger.info(
            "Streaming PC correction complete: all %d markers corrected",
            n_total_variants,
        )

    # Compute final post-correction diagnostic metrics if requested
    post_metrics: dict | None = None
    if diagnostic_marker_ids is not None and _diag_n_total > 0:
        # LRR_SD = sqrt(E[X²] - E[X]²) (population std, matching nanstd ddof=0)
        with np.errstate(invalid="ignore"):
            mean = _diag_sum / np.maximum(_diag_count, 1)
            mean_sq = _diag_sum_sq / np.maximum(_diag_count, 1)
            var = mean_sq - mean ** 2
            # Guard against negative variance from floating-point rounding
            var = np.maximum(var, 0.0)
            sd = np.sqrt(var)
        lrr_sd: list[float | None] = [
            None if _diag_count[j] == 0 else float(sd[j])
            for j in range(n_samples)
        ]
        callrate = [
            float(_diag_count[j]) / _diag_n_total
            for j in range(n_samples)
        ]
        post_metrics = {
            "SAMPLE": list(samples),
            "LRR_SD": lrr_sd,
            "callrate": callrate,
            "n_markers_used": _diag_count.tolist(),
        }
        logger.info(
            "Computed post-correction diagnostic metrics on %d markers "
            "during streaming",
            _diag_n_total,
        )

    return all_variants, total_skipped, post_metrics


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

_CORRECTION_HEADER_KEY = "batch_lrr_correction"


def _build_header_description(info: dict) -> str:
    """Build a human-readable description of the correction for the VCF header."""
    parts = [
        "LRR batch-effect correction applied by array_lrr_gwas.",
        f"Components removed (k): {info['k']}.",
        f"Components computed: {info.get('n_components_computed', info['k'])}.",
        f"Decomposition backend: {info['backend']}.",
        f"HQ samples used: {info['n_hq_samples']}.",
        f"Markers used: {info['n_markers_used']}.",
        f"Singular values (all computed): {np.array2string(info['singular_values'], precision=4, separator=',')}.",
        f"Date: {datetime.datetime.now(datetime.timezone.utc).isoformat()}.",
    ]
    return " ".join(parts)


def write_corrected(
    path_out: str | Path,
    corrected_lrr: NDArray[np.floating],
    samples: Sequence[str],
    variants: Sequence[dict],
    info: dict,
    *,
    path_template: str | Path | None = None,
) -> None:
    """Write batch-corrected LRR values to a BCF or VCF file.

    The output file contains the same variants and samples as the input,
    with the ``FORMAT/LRR`` field replaced by corrected values.  A
    structured header line documents the correction parameters.

    Parameters
    ----------
    path_out : str or Path
        Destination file path.  The extension determines the format:
        ``.bcf`` for BCF, anything else for VCF.
    corrected_lrr : ndarray, shape (n_variants, n_samples)
    samples : sequence of str
    variants : sequence of dict (as returned by :func:`read_lrr`).
    info : dict
        Correction metadata (from :func:`~array_lrr_gwas.correction.correct_lrr`).
    path_template : str or Path or None
        Optional input BCF/VCF used as a header template (preserves contigs, etc.).
    """
    path_out = str(path_out)
    is_bcf = path_out.endswith(".bcf")

    # Build header
    if path_template is not None:
        template = pysam.VariantFile(str(path_template))
        hdr = template.header.copy()
        template.close()
    else:
        hdr = pysam.VariantHeader()

    # Ensure LRR FORMAT exists
    if "LRR" not in hdr.formats:
        hdr.add_meta(
            "FORMAT",
            items=[
                ("ID", "LRR"),
                ("Number", "1"),
                ("Type", "Float"),
                ("Description", "Log R Ratio"),
            ],
        )

    # Ensure GT FORMAT exists (required by VCF spec)
    if "GT" not in hdr.formats:
        hdr.add_meta(
            "FORMAT",
            items=[
                ("ID", "GT"),
                ("Number", "1"),
                ("Type", "String"),
                ("Description", "Genotype"),
            ],
        )

    # Add correction description
    desc = _build_header_description(info)
    hdr.add_meta(key=_CORRECTION_HEADER_KEY, value=desc)

    # Add samples
    for s in samples:
        if s not in hdr.samples:
            hdr.add_sample(s)

    # Ensure contigs present
    for var in variants:
        chrom = var["chrom"]
        if chrom not in hdr.contigs:
            hdr.add_meta("contig", items=[("ID", chrom)])

    mode = "wb" if is_bcf else "w"
    vcf_out = pysam.VariantFile(path_out, mode, header=hdr)

    for i, var in enumerate(variants):
        # pysam requires at least 2 alleles (REF + ALT).  Intensity-only
        # markers have no ALT, so supply the VCF missing placeholder ".".
        alts = tuple(var.get("alts", ()))
        if not alts:
            alts = (".",)
        rec = vcf_out.new_record(
            contig=var["chrom"],
            start=var["pos"] - 1,  # pysam uses 0-based
            stop=var["pos"],
            alleles=(var["ref"],) + alts,
            id=var.get("id"),
        )
        for j, sample_data in enumerate(rec.samples.values()):
            val = corrected_lrr[i, j]
            if np.isnan(val):
                sample_data["LRR"] = None
            else:
                sample_data["LRR"] = float(val)
        vcf_out.write(rec)

    vcf_out.close()
