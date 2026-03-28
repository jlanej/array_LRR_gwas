"""Extract hard genotype calls from BCF/VCF files.

Reads the ``FORMAT/GT`` field and converts diploid calls to additive
dosages (0, 1, 2) representing the number of alternate alleles.
Missing or incomplete calls are encoded as ``np.nan``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

try:
    import pysam
except ImportError as _exc:
    raise ImportError(
        "pysam is required for BCF/VCF I/O. Install with: pip install pysam"
    ) from _exc


def read_genotypes(
    path: str | Path,
    *,
    min_maf: float = 0.01,
    min_call_rate: float = 0.90,
) -> tuple[NDArray[np.floating], list[str], list[dict]]:
    """Read genotype dosages from a BCF/VCF file.

    Diploid GT values are converted to additive dosages:
    ``0/0 → 0``, ``0/1 → 1``, ``1/1 → 2``.  Missing or haploid
    calls yield ``np.nan``.

    Parameters
    ----------
    path : str or Path
        Path to a BCF or VCF file containing ``FORMAT/GT``.
    min_maf : float
        Minimum minor allele frequency.  Variants below this
        threshold are excluded.
    min_call_rate : float
        Minimum genotype call rate.  Variants below this threshold
        are excluded.

    Returns
    -------
    dosage : ndarray, shape (n_variants, n_samples)
        Additive dosage matrix; missing entries are ``np.nan``.
    samples : list of str
        Sample identifiers in column order.
    variants : list of dict
        Per-variant metadata (``chrom``, ``pos``, ``id``, ``ref``,
        ``alts``).
    """
    path = str(path)
    vcf_in = pysam.VariantFile(path)
    samples = list(vcf_in.header.samples)
    n_samples = len(samples)

    dosage_rows: list[NDArray] = []
    variants: list[dict] = []

    for rec in vcf_in:
        row = np.full(n_samples, np.nan, dtype=np.float64)
        for i, sname in enumerate(samples):
            gt = rec.samples[sname].get("GT")
            if gt is None:
                continue
            alleles = gt
            if isinstance(alleles, tuple) and len(alleles) == 2:
                a0, a1 = alleles
                if a0 is not None and a1 is not None:
                    row[i] = float(a0 + a1)

        # Call rate filter
        valid = ~np.isnan(row)
        call_rate = valid.sum() / n_samples
        if call_rate < min_call_rate:
            continue

        # MAF filter
        called = row[valid]
        if len(called) == 0:
            continue
        freq = called.mean() / 2.0
        maf = min(freq, 1.0 - freq)
        if maf < min_maf:
            continue

        dosage_rows.append(row)
        variants.append({
            "chrom": rec.chrom,
            "pos": rec.pos,
            "id": rec.id,
            "ref": rec.ref,
            "alts": tuple(rec.alts) if rec.alts else (),
        })

    vcf_in.close()

    if not dosage_rows:
        dosage = np.empty((0, n_samples), dtype=np.float64)
    else:
        dosage = np.vstack(dosage_rows)

    return dosage, samples, variants
