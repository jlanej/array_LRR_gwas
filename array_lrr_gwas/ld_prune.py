"""LD pruning for GRM marker selection.

Provides sliding-window LD pruning to prevent highly correlated genomic
regions from disproportionately dominating the GRM eigenstructure.  Two
backends are available:

- **plink2** (default in CLI): Shells out to ``plink2 --indep-pairwise``
  for faster pruning on large datasets.  Requires ``plink2`` on
  ``$PATH``.
- **numpy**: Pure-Python greedy r²-based pruning.  Used when explicitly
  selected or as a fallback if plink2 is unavailable.

Default parameters (``window_bp=1_000_000``, ``r2_thresh=0.2``) follow
established best practice for GRM estimation (Yang *et al.* 2011, GCTA;
Privé *et al.* 2020, *Bioinformatics*).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure-NumPy backend
# ---------------------------------------------------------------------------

def _r2_vec(
    x: NDArray[np.floating],
    block: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Squared Pearson correlation of *x* with every row of *block*.

    Both *x* and *block* rows must already be centred (zero mean) and
    mean-imputed.  Returns an array of shape ``(block.shape[0],)`` with
    values in [0, 1].
    """
    # x: (n,)   block: (k, n)
    ss_x = x @ x
    if ss_x == 0:
        return np.zeros(block.shape[0])
    ss_block = np.sum(block * block, axis=1)
    cov = block @ x
    denom = ss_x * ss_block
    # Guard against zero-variance rows (should be rare after MAF filter)
    safe = denom > 0
    r2 = np.zeros(block.shape[0])
    r2[safe] = (cov[safe] ** 2) / denom[safe]
    return r2


def ld_prune(
    dosage: NDArray[np.floating],
    positions: NDArray[np.intp] | None = None,
    chromosomes: Sequence[str] | NDArray | None = None,
    *,
    window_bp: int = 1_000_000,
    r2_thresh: float = 0.2,
) -> NDArray[np.bool_]:
    """Return a boolean keep-mask after greedy LD pruning.

    Uses a forward-greedy algorithm: for each retained variant *i* in
    genomic order, all subsequent variants *j* within ``window_bp`` that
    have r²(i, j) > ``r2_thresh`` are removed.

    Parameters
    ----------
    dosage : ndarray, shape (n_variants, n_samples)
        Additive dosage matrix (0/1/2 with possible ``np.nan``).
    positions : ndarray of int, shape (n_variants,), optional
        Base-pair positions.  When provided together with *chromosomes*,
        the window is defined in bp.  Otherwise a flat variant-count
        window of 1 000 variants is used.
    chromosomes : array-like of str, shape (n_variants,), optional
        Chromosome labels.  Variants on different chromosomes are never
        pruned against each other.
    window_bp : int
        Maximum distance (in bp) between two variants for LD comparison
        (default: 1 000 000, i.e. 1 Mb).
    r2_thresh : float
        Variant pairs with r² above this value are LD-pruned
        (default: 0.2).

    Returns
    -------
    keep : ndarray of bool, shape (n_variants,)
        ``True`` for variants that survive LD pruning.
    """
    n_variants, n_samples = dosage.shape

    if n_variants == 0:
        return np.empty(0, dtype=bool)

    # Mean-impute & centre per variant
    Z = dosage.copy()
    nan_mask = np.isnan(Z)
    if nan_mask.any():
        row_means = np.nanmean(Z, axis=1, keepdims=True)
        Z = np.where(nan_mask, row_means, Z)
    Z -= Z.mean(axis=1, keepdims=True)

    have_coords = positions is not None and chromosomes is not None
    if have_coords:
        chroms = np.asarray(chromosomes, dtype=str)
        pos = np.asarray(positions)

    keep = np.ones(n_variants, dtype=bool)

    for i in range(n_variants):
        if not keep[i]:
            continue

        # Determine window of candidate variants to check
        if have_coords:
            # Same chromosome, within window_bp
            same_chrom = chroms == chroms[i]
            in_window = (
                same_chrom
                & (pos > pos[i])
                & (pos <= pos[i] + window_bp)
                & keep
            )
            # Only look at variants after i
            in_window[:i + 1] = False
        else:
            # Flat variant-count window (1000 variants)
            end = min(i + 1001, n_variants)
            in_window = np.zeros(n_variants, dtype=bool)
            in_window[i + 1:end] = keep[i + 1:end]

        j_indices = np.flatnonzero(in_window)
        if len(j_indices) == 0:
            continue

        r2 = _r2_vec(Z[i], Z[j_indices])
        prune_mask = r2 > r2_thresh
        keep[j_indices[prune_mask]] = False

    return keep


# ---------------------------------------------------------------------------
# plink2 backend
# ---------------------------------------------------------------------------

_FALLBACK_WINDOW_VARIANTS = 1000


def _plink2_available() -> bool:
    """Return True if plink2 is on PATH."""
    return shutil.which("plink2") is not None


def ld_prune_plink2(
    input_path: str | Path,
    *,
    window_kb: int = 1000,
    step: int = 50,
    r2_thresh: float = 0.2,
    min_maf: float = 0.01,
) -> set[str]:
    """LD-prune using ``plink2 --indep-pairwise``.

    Returns a set of variant IDs that survive pruning.

    Parameters
    ----------
    input_path : str or Path
        Path to BCF/VCF with ``FORMAT/GT``, **or** the prefix of a plink2
        BED/BIM/FAM fileset (with or without the ``.bed`` extension).
        When a BED fileset is provided, plink2 uses ``--bfile`` and skips
        the ``--maf`` re-filter (variants were already QC-filtered during
        BED generation).
    window_kb : int
        Window size in kilobases (default: 1000).
    step : int
        Step size in variants (default: 50).
    r2_thresh : float
        r² threshold for pruning (default: 0.2).
    min_maf : float
        Minimum MAF filter applied when reading BCF/VCF input (default: 0.01).
        Ignored when *input_path* points to a BED fileset.

    Returns
    -------
    keep_ids : set of str
        Variant IDs that survive pruning.

    Raises
    ------
    FileNotFoundError
        If plink2 is not found on PATH.
    subprocess.CalledProcessError
        If plink2 exits with a non-zero status.
    """
    if not _plink2_available():
        raise FileNotFoundError(
            "plink2 not found on PATH. Install plink2 or use "
            "the 'numpy' LD-pruning backend."
        )

    input_path = Path(input_path)
    suffix = input_path.suffix.lower()

    with tempfile.TemporaryDirectory() as tmp:
        out_prefix = str(Path(tmp) / "prune")
        cmd: list[str] = ["plink2"]

        if suffix == ".bed":
            # plink1 BED fileset — strip extension to get prefix
            bfile_prefix = str(input_path.with_suffix(""))
            cmd += ["--bfile", bfile_prefix]
        elif suffix == ".bcf":
            cmd += ["--bcf", str(input_path)]
            cmd += ["--maf", str(min_maf)]
        else:
            cmd += ["--vcf", str(input_path)]
            cmd += ["--maf", str(min_maf)]

        cmd += [
            "--allow-extra-chr",
            "--indep-pairwise", f"{window_kb}kb", str(step), str(r2_thresh),
            "--out", out_prefix,
        ]

        logger.info("Running plink2 LD pruning: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        prune_in = Path(f"{out_prefix}.prune.in")
        keep_ids: set[str] = set()
        if prune_in.exists():
            with open(prune_in) as fh:
                for line in fh:
                    vid = line.strip()
                    if vid:
                        keep_ids.add(vid)
        logger.info("plink2 LD pruning retained %d variants", len(keep_ids))
        return keep_ids
