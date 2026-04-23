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

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is a required dependency
    def tqdm(x, **_kwargs):  # type: ignore[no-redef]
        return x


def read_genotypes(
    path: str | Path,
    *,
    min_maf: float = 0.01,
    min_call_rate: float = 0.90,
    region: str | None = None,
    total_variants: int | None = None,
    progress: bool = True,
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
    region : str, optional
        If provided, restrict parsing to this region (e.g. ``"chrX"``
        or ``"chrX:1-155270560"``) using the BCF/VCF index. This avoids
        sequentially scanning autosomes when only a single contig is
        needed (e.g. for X-GRM computation), which can reduce runtime
        from hours to seconds on whole-genome BCFs.
    total_variants : int, optional
        Expected number of records iterated. Used only to display an
        accurate percentage in the progress bar.
    progress : bool
        If True (default), show a ``tqdm`` progress bar while parsing
        records. Set to False to suppress (e.g. in non-interactive
        logs or tests).

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

    # Use the BCF/VCF index to jump directly to ``region`` when given,
    # avoiding a full-genome sequential scan. Falls back to iterating
    # the whole file when ``region`` is None.
    if region is not None:
        try:
            record_iterator = vcf_in.fetch(region)
        except (ValueError, OSError) as exc:
            vcf_in.close()
            raise ValueError(
                f"Could not fetch region {region!r} from {path!r}. "
                "Ensure the file is indexed (.csi or .tbi) and the "
                "contig name matches the file's header."
            ) from exc
    else:
        record_iterator = vcf_in

    if progress:
        desc = f"Parsing genotypes ({region})" if region else "Parsing genotypes"
        record_iterator = tqdm(
            record_iterator,
            desc=desc,
            total=total_variants,
            unit=" vars",
        )

    for rec in record_iterator:
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
