"""BCF / VCF I/O utilities for batch-corrected LRR.

Reads LRR values from a BCF/VCF ``FORMAT/LRR`` field, applies batch-effect
correction via :func:`array_lrr_gwas.correction.correct_lrr`, and writes a
new BCF/VCF with the corrected values and a descriptive header entry.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

try:
    import pysam
except ImportError as _exc:
    raise ImportError(
        "pysam is required for BCF/VCF I/O. Install with: pip install pysam"
    ) from _exc


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
        for i, sname in enumerate(samples):
            val = rec.samples[sname].get("LRR")
            if val is not None:
                try:
                    row[i] = float(val)
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
        rec = vcf_out.new_record(
            contig=var["chrom"],
            start=var["pos"] - 1,  # pysam uses 0-based
            stop=var["pos"],
            alleles=(var["ref"],) + tuple(var.get("alts", ())),
            id=var.get("id"),
        )
        for j, sname in enumerate(samples):
            val = corrected_lrr[i, j]
            if np.isnan(val):
                rec.samples[sname]["LRR"] = None
            else:
                rec.samples[sname]["LRR"] = float(val)
        vcf_out.write(rec)

    vcf_out.close()
