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

For the **X-chromosome GRM (X-GRM)**, males are hemizygous and must be
coded as 0/2 (not 0/1) so that their dosage maps onto the female scale.
PAR/XTR regions are excluded because they recombine like autosomes.
See :func:`compute_x_grm` for the full implementation.

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
import time
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


def compute_x_grm(
    dosage: NDArray[np.floating],
    is_male: NDArray[np.bool_],
    *,
    variant_positions: list[tuple[str, int]] | None = None,
    par_regions: dict[str, list[tuple[int, int]]] | None = None,
    min_maf: float = 0.01,
) -> NDArray[np.floating]:
    """Compute the X-chromosome Genetic Relationship Matrix (X-GRM).

    Implements the GCTA X-GRM mathematics (Yang et al. 2011) with
    sex-aware genotype coding:

    * **Males** (hemizygous): dosage coded as 0 or 2 (not 0/1).
    * **Females** (diploid): dosage coded as 0, 1, or 2.
    * **PAR/XTR regions** are excluded (autosomal-like recombination).
    * Allele frequencies are computed jointly across all samples
      using the 0/2 male coding.
    * Standardisation uses the autosomal formula which correctly
      yields variance = 2 for males (complete hemizygosity).

    Parameters
    ----------
    dosage : ndarray, shape (n_variants, n_samples)
        Additive dosage matrix for non-PAR chrX variants.
        Males **should** already be coded as 0/2 from the BCF/VCF;
        if 0/1 coding is detected it is automatically rescaled.
    is_male : ndarray of bool, shape (n_samples,)
        ``True`` for male samples, ``False`` for female samples.
    variant_positions : list of (chrom, pos) or None
        Per-variant chromosome and position for PAR exclusion.
        When ``None``, no PAR filtering is performed (caller is
        responsible for pre-filtering).
    par_regions : dict mapping chrom → list of (start, end) or None
        PAR/XTR regions to exclude.  Only used when
        *variant_positions* is also provided.
    min_maf : float
        Minimum minor allele frequency.  Variants below this
        threshold are excluded.

    Returns
    -------
    grm : ndarray, shape (n_samples, n_samples)
        Symmetric X-chromosome GRM.

    Raises
    ------
    ValueError
        If no variants remain after filtering, or dimension mismatches.
    """
    n_variants, n_samples = dosage.shape
    if is_male.shape != (n_samples,):
        raise ValueError(
            f"is_male must have shape ({n_samples},), got {is_male.shape}"
        )

    _t0 = time.monotonic()
    n_males = int(is_male.sum())
    n_females = n_samples - n_males
    logger.info(
        "X-GRM: starting computation — %d variants × %d samples "
        "(%d males, %d females)",
        n_variants, n_samples, n_males, n_females,
    )

    Z = dosage.copy()

    # --- PAR/XTR exclusion ---
    if variant_positions is not None and par_regions is not None:
        par_mask = np.zeros(n_variants, dtype=bool)
        for i, (chrom, pos) in enumerate(variant_positions):
            regions = par_regions.get(chrom, [])
            for start, end in regions:
                if start <= pos <= end:
                    par_mask[i] = True
                    break
        n_par = int(par_mask.sum())
        if n_par > 0:
            logger.info(
                "X-GRM: excluding %d PAR/XTR variants (%d remain)",
                n_par, n_variants - n_par,
            )
            Z = Z[~par_mask]
            n_variants = Z.shape[0]

    if n_variants == 0:
        raise ValueError(
            "No X-chromosome variants remain after PAR/XTR exclusion; "
            "cannot compute X-GRM."
        )

    # --- Male dosage rescaling (0/1 → 0/2) ---
    # Males are hemizygous on chrX, so their genotype dosage should be
    # 0 or 2 (not 0/1).  If any male value is near 1.0 (heterozygous
    # call) *and* the maximum male dosage is ≤ 1, assume the input uses
    # 0/1 coding and rescale to 0/2.  The max-dosage check prevents
    # corrupting data that is already on the 0/2 scale but contains a
    # noisy imputed value near 1.0.
    _MALE_HET_DETECT_TOL = 0.05
    _MALE_MAX_DOSAGE_01_THRESHOLD = 1.05
    male_cols = np.where(is_male)[0]
    if male_cols.size > 0:
        male_data = Z[:, male_cols]
        finite_male = male_data[np.isfinite(male_data)]
        if finite_male.size > 0:
            has_het = np.any(np.abs(finite_male - 1.0) < _MALE_HET_DETECT_TOL)
            if has_het:
                if np.max(finite_male) <= _MALE_MAX_DOSAGE_01_THRESHOLD:
                    logger.info(
                        "X-GRM: rescaling male dosages from 0/1 to 0/2 coding"
                    )
                    Z[:, male_cols] = male_data * 2.0
                else:
                    logger.debug(
                        "X-GRM: detected male dosages near 1.0, but max "
                        "dosage (%.2f) suggests 0/2 coding; skipping rescale",
                        np.max(finite_male),
                    )

    # --- Mean-impute missing values per variant ---
    nan_mask = np.isnan(Z)
    if nan_mask.any():
        row_means = np.nanmean(Z, axis=1, keepdims=True)
        Z = np.where(nan_mask, row_means, Z)

    # --- Joint allele frequency: p_j = sum(x_ij) / (2N) ---
    freq = Z.mean(axis=1) / 2.0
    maf = np.minimum(freq, 1.0 - freq)

    # --- MAF filter ---
    n_before_maf = n_variants
    keep = maf >= min_maf
    Z = Z[keep]
    freq = freq[keep]

    if Z.shape[0] == 0:
        raise ValueError(
            "No X-chromosome variants remain after MAF filtering "
            f"(min_maf={min_maf}); cannot compute X-GRM."
        )

    m = Z.shape[0]
    logger.info(
        "X-GRM MAF filter (min_maf=%.4f): %d → %d variants",
        min_maf, n_before_maf, m,
    )

    # --- Standardise: z_ij = (x_ij - 2p_j) / sqrt(2 p_j (1 - p_j)) ---
    denom = np.sqrt(2.0 * freq * (1.0 - freq))
    safe = denom > 0
    Z[safe] = (Z[safe] - 2.0 * freq[safe, np.newaxis]) / denom[safe, np.newaxis]
    Z[~safe] = 0.0

    # --- X-GRM = (1/M) Z Z^T ---
    logger.info(
        "X-GRM: computing %d × %d matrix product (%d variants × %d samples) …",
        n_samples, n_samples, m, n_samples,
    )
    _t_matmul = time.monotonic()
    grm = (Z.T @ Z) / m
    _elapsed_matmul = time.monotonic() - _t_matmul
    _elapsed_total = time.monotonic() - _t0

    logger.info(
        "X-GRM computed: %d × %d from %d non-PAR X-chromosome variants "
        "(%d males, %d females) — matmul %.1f s, total %.1f s",
        grm.shape[0], grm.shape[1], m,
        int(is_male.sum()), int((~is_male).sum()),
        _elapsed_matmul, _elapsed_total,
    )

    return grm


def _plink2_available() -> bool:
    return shutil.which("plink2") is not None


def _read_bim_variant_ids(bim_path: str | Path) -> list[str]:
    """Read variant IDs (column 2) from a plink1 BIM file.

    BIM columns: chrom, variant_id, cm, pos, alt, ref

    Raises
    ------
    FileNotFoundError
        If *bim_path* does not exist.
    ValueError
        If *bim_path* is empty or contains no valid variant IDs.
    """
    bim_path = Path(bim_path)
    if not bim_path.exists():
        raise FileNotFoundError(
            f"plink2 BIM file not found: {bim_path}. "
            "Ensure that make_plink2_bed() completed successfully."
        )
    ids: list[str] = []
    with open(bim_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                ids.append(parts[1])
    return ids


def _plink2_input_args(input_path: Path) -> list[str]:
    """Return the plink2 input flag(s) for *input_path*.

    Supports BCF, VCF, and plink1 BED filesets.  When a ``.bed``, ``.bim``,
    or ``.fam`` suffix is detected, the prefix (without extension) is used
    with ``--bfile``.
    """
    suffix = input_path.suffix.lower()
    if suffix in (".bed", ".bim", ".fam"):
        return ["--bfile", str(input_path.with_suffix(""))]
    if suffix == ".bcf":
        return ["--bcf", str(input_path)]
    return ["--vcf", str(input_path)]


def make_plink2_bed(
    input_path: str | Path,
    out_prefix: str | Path,
    *,
    keep_variants: list[str] | None = None,
    keep_samples: list[str] | None = None,
    min_maf: float = 0.01,
    min_call_rate: float = 0.90,
    allow_extra_chr: bool = True,
    autosome_only: bool = True,
) -> Path:
    """Convert a BCF/VCF to plink2 BED/BIM/FAM using plink2.

    Optionally filters to a pre-selected set of variant IDs (e.g. those
    passing upstream QC) and/or a sample ID list.  Writes
    ``{out_prefix}.bed``, ``{out_prefix}.bim``, ``{out_prefix}.fam``.

    Parameters
    ----------
    input_path : str or Path
        Input BCF/VCF with ``FORMAT/GT``.  Also accepts a plink1 BED
        fileset prefix or ``.bed`` path when further filtering a BED.
    out_prefix : str or Path
        Output prefix for plink2 binary files.
    keep_variants : list of str, optional
        Variant IDs to retain.  Written to a temporary extract file.
        When *None* and ``min_maf`` > 0, plink2 applies its own MAF filter.
    keep_samples : list of str, optional
        Sample IDs to retain.  Written to a temporary keep file.
    min_maf : float
        Minimum minor allele frequency.  Set to ``0.0`` when *keep_variants*
        already represents a QC-filtered set to avoid secondary filtering.
    min_call_rate : float
        Minimum per-variant call rate (``1 - missing rate``).
    allow_extra_chr : bool
        Pass ``--allow-extra-chr`` to plink2 (required for non-human contigs).
    autosome_only : bool
        Pass ``--autosome`` and ``--max-alleles 2`` to plink2.  This restricts
        the BED to biallelic autosomal variants, which avoids sex-chromosome
        handling requirements (sex info needed for chrX/Y) and multiallelic
        variant errors.  Appropriate for GRM computation.  Defaults to ``True``.

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

    input_path = Path(input_path)
    out_prefix = Path(out_prefix)

    with tempfile.TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)
        cmd: list[str] = ["plink2"]
        cmd += _plink2_input_args(input_path)

        if allow_extra_chr:
            cmd.append("--allow-extra-chr")

        if autosome_only:
            cmd += ["--autosome", "--max-alleles", "2"]

        if min_maf > 0.0:
            cmd += ["--maf", str(min_maf)]
        cmd += ["--geno", str(1.0 - min_call_rate)]

        if keep_variants:
            extract_file = tmp / "extract.txt"
            extract_file.write_text("\n".join(keep_variants) + "\n")
            cmd += ["--extract", str(extract_file)]

        if keep_samples:
            keep_file = tmp / "keep.txt"
            keep_file.write_text(
                "#IID\n" + "\n".join(keep_samples) + "\n"
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
    input_path: str | Path,
    *,
    keep_variants: set[str] | list[str] | None = None,
    keep_samples: list[str] | None = None,
    min_maf: float = 0.01,
) -> tuple[NDArray[np.floating], list[str]]:
    """Compute the GRM using ``plink2 --make-grm-bin``.

    This calls ``plink2`` to generate the lower-triangle GRM in
    GCTA binary format (``.grm.bin``, ``.grm.id``, ``.grm.N.bin``),
    then reads and converts it to a symmetric NumPy matrix.

    Parameters
    ----------
    input_path : str or Path
        Path to BCF/VCF with ``FORMAT/GT``, **or** the prefix of a plink1
        BED fileset (with or without the ``.bed`` extension).  Passing a
        BED fileset is preferred: variants are read directly without
        re-parsing the BCF.
    keep_variants : set or list of str, optional
        Variant IDs to restrict GRM computation (e.g. LD-pruned set).
        Written to a temporary ``--extract`` file.  When *None* all
        variants in *input_path* are used.
    keep_samples : list of str, optional
        Sample IDs to restrict GRM computation.  When *input_path* is a
        BED already filtered to the desired samples this can be omitted.
    min_maf : float
        Minimum MAF filter applied by plink2 when reading BCF/VCF input.
        Ignored when *input_path* is a BED fileset.

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

    input_path = Path(input_path)
    is_bed = input_path.suffix.lower() == ".bed"

    with tempfile.TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)
        out_prefix = tmp / "grm"
        cmd: list[str] = ["plink2"]
        cmd += _plink2_input_args(input_path)
        cmd += ["--allow-extra-chr"]

        # Apply MAF filter only for BCF/VCF input; BED already QC-filtered.
        if not is_bed and min_maf > 0.0:
            cmd += ["--maf", str(min_maf)]

        if keep_variants:
            ext_file = tmp / "extract.txt"
            ext_file.write_text("\n".join(keep_variants) + "\n")
            cmd += ["--extract", str(ext_file)]

        if keep_samples:
            keep_file = tmp / "keep.txt"
            keep_file.write_text(
                "#IID\n" + "\n".join(keep_samples) + "\n"
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

