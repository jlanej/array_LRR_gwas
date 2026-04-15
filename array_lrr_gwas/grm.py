"""Genetic Relationship Matrix (GRM) computation.

Computes a sample-by-sample GRM from an additive dosage matrix using
the standard allele-frequency-standardised estimator (Yang *et al.*
2011, *Nat Genet* 43:519–525).

The GRM is defined as:

.. math::

    \\hat{K} = \\frac{1}{M} Z Z^\\top

where *Z* is the *n × M* matrix of standardised genotypes:

.. math::

    z_{ij} = \\frac{x_{ij} - 2 p_j}{\\sqrt{2 p_j (1 - p_j)}}

Missing genotypes are mean-imputed *before* standardisation.

.. note::

   For best-practice GRM estimation, the input dosage matrix should be
   **LD-pruned** prior to calling :func:`compute_grm`.  Highly linked
   regions can otherwise disproportionately dominate the GRM
   eigenstructure.  See :mod:`array_lrr_gwas.ld_prune` for a
   sliding-window r²-based pruning implementation, and the CLI
   ``--no-ld-prune`` / ``--ld-r2-thresh`` flags on the ``associate``
   sub-command.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def compute_grm(
    dosage: NDArray[np.floating],
    *,
    min_maf: float = 0.01,
) -> NDArray[np.floating]:
    """Compute the genomic relationship matrix.

    Parameters
    ----------
    dosage : ndarray, shape (n_variants, n_samples)
        Additive dosage matrix (0/1/2 with possible ``np.nan``).
    min_maf : float
        Variants with MAF below this threshold are excluded from the
        GRM calculation.

    Returns
    -------
    grm : ndarray, shape (n_samples, n_samples)
        Symmetric, positive-semidefinite GRM.

    Raises
    ------
    ValueError
        If no variants remain after MAF filtering.
    """
    n_variants, n_samples = dosage.shape

    # Mean-impute missing values per variant (vectorised)
    Z = dosage.copy()
    nan_mask = np.isnan(Z)
    if nan_mask.any():
        row_means = np.nanmean(Z, axis=1, keepdims=True)
        Z = np.where(nan_mask, row_means, Z)

    # Compute allele frequencies
    freq = Z.mean(axis=1) / 2.0
    maf = np.minimum(freq, 1.0 - freq)

    # MAF filter
    keep = maf >= min_maf
    Z = Z[keep]
    freq = freq[keep]

    if Z.shape[0] == 0:
        raise ValueError(
            "No variants remain after MAF filtering "
            f"(min_maf={min_maf}); cannot compute GRM."
        )

    m = Z.shape[0]

    # Standardise: z_ij = (x_ij - 2p_j) / sqrt(2 p_j (1 - p_j))
    denom = np.sqrt(2.0 * freq * (1.0 - freq))
    safe = denom > 0
    Z[safe] = (Z[safe] - 2.0 * freq[safe, np.newaxis]) / denom[safe, np.newaxis]
    Z[~safe] = 0.0  # monomorphic after imputation

    # GRM = (1/M) Z^T Z  (Z is m×n, we want n×n)
    grm = (Z.T @ Z) / m
    return grm


def _plink2_available() -> bool:
    return shutil.which("plink2") is not None


def make_plink2_bed(
    bcf_path: str | Path,
    out_prefix: str | Path,
    *,
    keep_variants: list[str] | None = None,
    keep_samples: list[str] | None = None,
    min_maf: float = 0.01,
    min_call_rate: float = 0.90,
    allow_extra_chr: bool = True,
) -> Path:
    """Convert a BCF/VCF to plink2 BED/BIM/FAM using plink2.

    Optionally filters to a pre-selected set of variant IDs (e.g. the
    LD-pruned marker set) and/or a sample ID list.  Writes
    ``{out_prefix}.bed``, ``{out_prefix}.bim``, ``{out_prefix}.fam``.

    Parameters
    ----------
    bcf_path : str or Path
        Input BCF/VCF with ``FORMAT/GT``.
    out_prefix : str or Path
        Output prefix for plink2 binary files.
    keep_variants : list of str, optional
        Variant IDs to retain.  Written to a temporary extract file.
    keep_samples : list of str, optional
        Sample IDs to retain.  Written to a temporary keep file.
    min_maf : float
        Minimum minor allele frequency.
    min_call_rate : float
        Minimum per-variant call rate (``1 - missing rate``).
    allow_extra_chr : bool
        Pass ``--allow-extra-chr`` to plink2 (required for non-human contigs).

    Returns
    -------
    bed_path : Path
        Path to the ``.bed`` file.

    Raises
    ------
    FileNotFoundError
        If plink2 is not on PATH.
    subprocess.CalledProcessError
        If plink2 exits with non-zero status.
    """
    if not _plink2_available():
        raise FileNotFoundError(
            "plink2 not found on PATH. Install plink2 "
            "(https://www.cog-genomics.org/plink/2.0/)."
        )

    bcf_path = Path(bcf_path)
    out_prefix = Path(out_prefix)
    suffix = bcf_path.suffix.lower()

    with tempfile.TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)
        cmd: list[str] = ["plink2"]

        if suffix == ".bcf":
            cmd += ["--bcf", str(bcf_path)]
        else:
            cmd += ["--vcf", str(bcf_path)]

        if allow_extra_chr:
            cmd.append("--allow-extra-chr")

        cmd += ["--maf", str(min_maf)]
        cmd += ["--geno", str(1.0 - min_call_rate)]

        if keep_variants:
            extract_file = tmp / "extract.txt"
            extract_file.write_text("\n".join(keep_variants) + "\n")
            cmd += ["--extract", str(extract_file)]

        if keep_samples:
            keep_file = tmp / "keep.txt"
            keep_file.write_text(
                "\n".join(f"{s}\t{s}" for s in keep_samples) + "\n"
            )
            cmd += ["--keep", str(keep_file)]

        cmd += ["--make-bed", "--out", str(out_prefix)]

        logger.info("Running plink2 BED generation: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, capture_output=True, text=True)

    bed_path = out_prefix.with_suffix(".bed")
    if not bed_path.exists():
        raise RuntimeError(
            f"plink2 BED generation did not produce {bed_path}"
        )
    return bed_path


def compute_grm_plink2(
    bcf_path: str | Path,
    *,
    keep_variants: list[str] | None = None,
    keep_samples: list[str] | None = None,
    min_maf: float = 0.01,
) -> tuple[NDArray[np.floating], list[str]]:
    """Compute the GRM using ``plink2 --make-grm-bin``.

    This calls ``plink2`` to generate the lower-triangle GRM in
    GCTA binary format (``.grm.bin``, ``.grm.id``, ``.grm.N.bin``),
    then reads and converts it to a symmetric NumPy matrix.

    Parameters
    ----------
    bcf_path : str or Path
        Input BCF/VCF with ``FORMAT/GT``.
    keep_variants : list of str, optional
        Variant IDs (LD-pruned set) to restrict GRM computation.
    keep_samples : list of str, optional
        Sample IDs to restrict GRM computation.
    min_maf : float
        Minimum MAF filter applied by plink2.

    Returns
    -------
    grm : ndarray, shape (n_samples, n_samples)
        Symmetric GRM.
    sample_ids : list of str
        Sample identifiers in the order used by plink2.

    Raises
    ------
    FileNotFoundError
        If plink2 is not on PATH.
    subprocess.CalledProcessError
        If plink2 exits with non-zero status.
    """
    if not _plink2_available():
        raise FileNotFoundError(
            "plink2 not found on PATH. Install plink2 "
            "(https://www.cog-genomics.org/plink/2.0/)."
        )

    bcf_path = Path(bcf_path)
    suffix = bcf_path.suffix.lower()

    with tempfile.TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)
        out_prefix = tmp / "grm"
        cmd: list[str] = ["plink2"]

        if suffix == ".bcf":
            cmd += ["--bcf", str(bcf_path)]
        else:
            cmd += ["--vcf", str(bcf_path)]

        cmd += ["--allow-extra-chr", "--maf", str(min_maf)]

        if keep_variants:
            ext_file = tmp / "extract.txt"
            ext_file.write_text("\n".join(keep_variants) + "\n")
            cmd += ["--extract", str(ext_file)]

        if keep_samples:
            keep_file = tmp / "keep.txt"
            keep_file.write_text(
                "\n".join(f"{s}\t{s}" for s in keep_samples) + "\n"
            )
            cmd += ["--keep", str(keep_file)]

        cmd += ["--make-grm-bin", "--out", str(out_prefix)]

        logger.info("Running plink2 GRM computation: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Read sample IDs from .grm.id (FID TAB IID)
        grm_id_path = out_prefix.with_suffix(".grm.id")
        sample_ids: list[str] = []
        with open(grm_id_path) as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                # Use IID (second column) as the sample identifier.
                sample_ids.append(parts[1] if len(parts) > 1 else parts[0])

        n = len(sample_ids)

        # Read lower-triangle GRM from .grm.bin (float32 values in row order)
        grm_bin_path = out_prefix.with_suffix(".grm.bin")
        n_pairs = n * (n + 1) // 2
        raw = np.fromfile(str(grm_bin_path), dtype=np.float32, count=n_pairs)
        # Fill symmetric matrix from lower triangle.
        grm = np.zeros((n, n), dtype=np.float64)
        idx = 0
        for i in range(n):
            for j in range(i + 1):
                grm[i, j] = raw[idx]
                grm[j, i] = raw[idx]
                idx += 1

    return grm, sample_ids

