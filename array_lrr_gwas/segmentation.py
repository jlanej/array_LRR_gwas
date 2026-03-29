"""Post-GWAS segmentation: define CNV-associated intervals from per-marker results.

Implements a two-state Hidden Markov Model (HMM) that partitions markers into
*associated* and *null* states, then collapses contiguous associated runs into
BED-format intervals with provenance statistics.

A simpler threshold-and-merge strategy is also available as a baseline.

See ``docs/segmentation_design.md`` for the full design rationale.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum p-value floor to avoid -inf in log-transform.
_P_FLOOR: float = 1e-300

# HMM defaults
_DEFAULT_NULL_RATE: float = np.log(10)  # theoretical rate for -log10(p) under H0
_DEFAULT_SIGNAL_RATE: float = 0.1       # associated state expects mean ~10
_DEFAULT_PRIOR_ASSOC: float = 1e-3      # 1 in 1000 markers expected associated
_DEFAULT_TRANSITION: float = 1e-4       # per-marker state-transition probability

# Threshold defaults
_DEFAULT_P_THRESHOLD: float = 5e-8
_DEFAULT_MAX_GAP: int = 1_000_000

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class SegmentationResult:
    """Collection of CNV-associated genomic intervals (BED-like).

    Attributes
    ----------
    chrom : list[str]
        Chromosome identifiers for each segment.
    start : list[int]
        0-based inclusive start positions (BED convention).
    end : list[int]
        0-based exclusive end positions (BED convention).
    name : list[str]
        Region identifiers (e.g. ``region_1``).
    n_markers : list[int]
        Number of contributing markers per segment.
    min_p : list[float]
        Minimum p-value within each segment.
    mean_beta : list[float]
        Mean effect-size estimate within each segment.
    max_abs_stat : list[float]
        Maximum absolute test statistic within each segment.
    method : list[str]
        Source association method for each segment.
    strategy : str
        Segmentation strategy (``"hmm"`` or ``"threshold"``).
    parameters : dict
        Strategy-specific parameters used for provenance.
    """

    chrom: list[str] = field(default_factory=list)
    start: list[int] = field(default_factory=list)
    end: list[int] = field(default_factory=list)
    name: list[str] = field(default_factory=list)
    n_markers: list[int] = field(default_factory=list)
    min_p: list[float] = field(default_factory=list)
    mean_beta: list[float] = field(default_factory=list)
    max_abs_stat: list[float] = field(default_factory=list)
    method: list[str] = field(default_factory=list)
    strategy: str = ""
    parameters: dict = field(default_factory=dict)

    # ---- serialisation helpers -------------------------------------------

    _BED_FIELDS: tuple[str, ...] = (
        "chrom", "start", "end", "name", "n_markers",
        "min_p", "mean_beta", "max_abs_stat", "method",
    )

    def to_records(self) -> list[dict[str, object]]:
        """One dict per segment with all BED+ columns."""
        records: list[dict[str, object]] = []
        for i in range(len(self.chrom)):
            records.append({
                "chrom": self.chrom[i],
                "start": self.start[i],
                "end": self.end[i],
                "name": self.name[i],
                "n_markers": self.n_markers[i],
                "min_p": float(self.min_p[i]),
                "mean_beta": float(self.mean_beta[i]),
                "max_abs_stat": float(self.max_abs_stat[i]),
                "method": self.method[i],
            })
        return records

    def write_bed(self, path: str | Path) -> None:
        """Write BED-format output with a header comment line.

        The header is prefixed with ``#`` and contains column names.
        One row per segment follows in tab-delimited format.
        """
        path = Path(path)
        records = self.to_records()
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=list(self._BED_FIELDS),
                delimiter="\t",
            )
            # Write comment-style header
            fh.write("#" + "\t".join(self._BED_FIELDS) + "\n")
            writer.writerows(records)
        logger.info("Wrote %d segments to %s", len(records), path)


# ---------------------------------------------------------------------------
# Reading association TSV (output of ``associate`` sub-command)
# ---------------------------------------------------------------------------


def read_association_tsv(path: str | Path) -> list[dict[str, object]]:
    """Read the per-marker association TSV written by the ``associate`` CLI.

    Returns a list of record dicts with the standard association-result
    schema (``chrom``, ``pos``, ``variant_id``, ``beta``, ``se``, ``stat``,
    ``p_value``, ``n_samples``, ``method``).

    Numeric fields are cast to their expected Python types.
    """
    path = Path(path)
    records: list[dict[str, object]] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            records.append({
                "chrom": row["chrom"],
                "pos": int(row["pos"]),
                "variant_id": row["variant_id"],
                "beta": float(row["beta"]),
                "se": float(row["se"]),
                "stat": float(row["stat"]),
                "p_value": float(row["p_value"]),
                "n_samples": int(row["n_samples"]),
                "method": row["method"],
            })
    return records


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def segment_associations(
    records: Sequence[dict[str, object]],
    *,
    strategy: str = "hmm",
    p_threshold: float = _DEFAULT_P_THRESHOLD,
    max_gap: int = _DEFAULT_MAX_GAP,
    min_markers: int = 1,
    null_rate: float = _DEFAULT_NULL_RATE,
    signal_rate: float = _DEFAULT_SIGNAL_RATE,
    prior_assoc: float = _DEFAULT_PRIOR_ASSOC,
    transition_prob: float = _DEFAULT_TRANSITION,
) -> SegmentationResult:
    """Segment per-marker GWAS results into CNV-associated intervals.

    Parameters
    ----------
    records
        List of dicts from ``AssociationResult.to_records()`` or
        ``read_association_tsv()``.  Each dict must contain at minimum
        ``chrom``, ``pos``, ``p_value``, ``beta``, ``stat``, ``method``.
    strategy
        ``"hmm"`` (default) for HMM-based segmentation, or ``"threshold"``
        for a simpler p-value threshold with distance-based merging.
    p_threshold
        (threshold strategy) Significance threshold (default 5e-8).
    max_gap
        (threshold strategy) Maximum gap in bp between markers to merge
        into a single segment (default 1 000 000).
    min_markers
        Minimum number of markers to keep a segment (default 1).
    null_rate
        (HMM strategy) Exponential rate for the null-state emission
        model on -log10(p). The theoretical value is ``ln(10)`` ≈ 2.303.
    signal_rate
        (HMM strategy) Exponential rate for the associated-state emission
        model on -log10(p). Smaller values expect stronger signals
        (default 0.1 → mean -log10(p) = 10).
    prior_assoc
        (HMM strategy) Prior probability a marker is in the associated
        state (default 0.001).
    transition_prob
        (HMM strategy) Per-marker state-transition probability
        (default 1e-4).

    Returns
    -------
    SegmentationResult
        BED-format intervals with summary statistics and provenance.
    """
    if strategy not in ("hmm", "threshold"):
        raise ValueError(f"Unknown strategy {strategy!r}; expected 'hmm' or 'threshold'")

    # Separate records by chromosome while preserving order.
    chrom_order: list[str] = []
    by_chrom: dict[str, list[dict[str, object]]] = {}
    for rec in records:
        c = str(rec["chrom"])
        if c not in by_chrom:
            chrom_order.append(c)
            by_chrom[c] = []
        by_chrom[c].append(rec)

    # Collect parameters for provenance.
    if strategy == "hmm":
        params = dict(
            null_rate=null_rate,
            signal_rate=signal_rate,
            prior_assoc=prior_assoc,
            transition_prob=transition_prob,
            min_markers=min_markers,
        )
    else:
        params = dict(
            p_threshold=p_threshold,
            max_gap=max_gap,
            min_markers=min_markers,
        )

    all_segments: list[_RawSegment] = []
    for chrom in chrom_order:
        recs = by_chrom[chrom]
        if strategy == "hmm":
            segs = _hmm_segment(recs, null_rate, signal_rate,
                                prior_assoc, transition_prob)
        else:
            segs = _threshold_segment(recs, p_threshold, max_gap)
        all_segments.extend(segs)

    # Filter by min_markers.
    all_segments = [s for s in all_segments if s.n_markers >= min_markers]

    # Build result.
    result = SegmentationResult(strategy=strategy, parameters=params)
    for idx, seg in enumerate(all_segments, start=1):
        result.chrom.append(seg.chrom)
        result.start.append(seg.start)
        result.end.append(seg.end)
        result.name.append(f"region_{idx}")
        result.n_markers.append(seg.n_markers)
        result.min_p.append(seg.min_p)
        result.mean_beta.append(seg.mean_beta)
        result.max_abs_stat.append(seg.max_abs_stat)
        result.method.append(seg.method)

    logger.info(
        "Segmentation (%s) produced %d regions from %d markers",
        strategy, len(result.chrom), len(records),
    )
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@dataclass
class _RawSegment:
    chrom: str
    start: int       # 0-based inclusive
    end: int         # 0-based exclusive
    n_markers: int
    min_p: float
    mean_beta: float
    max_abs_stat: float
    method: str


def _collect_segments(
    records: Sequence[dict[str, object]],
    mask: Sequence[bool],
) -> list[_RawSegment]:
    """Collapse runs of True in *mask* into ``_RawSegment`` objects.

    Positions in *records* are 1-based (GWAS convention).  BED output uses
    0-based half-open coordinates, so ``start = pos - 1`` and ``end = pos``
    for the bounding markers.
    """
    segments: list[_RawSegment] = []
    n = len(records)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        # Start of a run.
        run_start = i
        while i < n and mask[i]:
            i += 1
        run_end = i  # exclusive
        run_recs = records[run_start:run_end]
        positions = [int(r["pos"]) for r in run_recs]
        p_values = [float(r["p_value"]) for r in run_recs]
        betas = [float(r["beta"]) for r in run_recs]
        stats_ = [float(r["stat"]) for r in run_recs]
        methods = [str(r["method"]) for r in run_recs]

        seg = _RawSegment(
            chrom=str(run_recs[0]["chrom"]),
            start=min(positions) - 1,            # 0-based
            end=max(positions),                   # 0-based exclusive
            n_markers=len(run_recs),
            min_p=min(p_values),
            mean_beta=float(np.mean(betas)),
            max_abs_stat=float(max(abs(s) for s in stats_)),
            method=methods[0],
        )
        segments.append(seg)
    return segments


# ---------------------------------------------------------------------------
# Threshold-and-merge strategy
# ---------------------------------------------------------------------------


def _threshold_segment(
    records: Sequence[dict[str, object]],
    p_threshold: float,
    max_gap: int,
) -> list[_RawSegment]:
    """Flag markers below *p_threshold*, then merge nearby flagged markers."""
    if not records:
        return []

    # Flag significant markers.
    mask = [float(r["p_value"]) < p_threshold for r in records]

    # Initial segments from contiguous flagged runs.
    raw = _collect_segments(records, mask)

    # Merge segments within max_gap on the same chromosome.
    merged: list[_RawSegment] = []
    for seg in raw:
        if (
            merged
            and merged[-1].chrom == seg.chrom
            and seg.start - merged[-1].end <= max_gap
        ):
            prev = merged[-1]
            # Merge: expand interval, combine stats.
            prev.end = seg.end
            old_n = prev.n_markers
            prev.n_markers += seg.n_markers
            prev.min_p = min(prev.min_p, seg.min_p)
            prev.mean_beta = (
                (prev.mean_beta * old_n + seg.mean_beta * seg.n_markers)
                / prev.n_markers
            )
            prev.max_abs_stat = max(prev.max_abs_stat, seg.max_abs_stat)
        else:
            merged.append(seg)

    return merged


# ---------------------------------------------------------------------------
# HMM strategy
# ---------------------------------------------------------------------------


def _viterbi_decode(
    log_emissions: np.ndarray,
    log_prior: np.ndarray,
    log_trans: np.ndarray,
) -> np.ndarray:
    """Viterbi algorithm for a two-state HMM.

    Parameters
    ----------
    log_emissions : ndarray, shape (T, 2)
        Log emission probabilities for each time step and state.
    log_prior : ndarray, shape (2,)
        Log prior probabilities for initial states.
    log_trans : ndarray, shape (2, 2)
        Log transition matrix.

    Returns
    -------
    states : ndarray, shape (T,), dtype int
        Optimal state sequence (0 = null, 1 = associated).
    """
    T = log_emissions.shape[0]
    if T == 0:
        return np.array([], dtype=int)

    # Forward pass: compute delta (best log-probability) and psi (backpointer).
    delta = np.empty((T, 2), dtype=np.float64)
    psi = np.empty((T, 2), dtype=int)

    delta[0] = log_prior + log_emissions[0]
    psi[0] = 0

    for t in range(1, T):
        for s in range(2):
            candidates = delta[t - 1] + log_trans[:, s]
            best = int(np.argmax(candidates))
            delta[t, s] = candidates[best] + log_emissions[t, s]
            psi[t, s] = best

    # Backtrack.
    states = np.empty(T, dtype=int)
    states[-1] = int(np.argmax(delta[-1]))
    for t in range(T - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]

    return states


def _hmm_segment(
    records: Sequence[dict[str, object]],
    null_rate: float,
    signal_rate: float,
    prior_assoc: float,
    transition_prob: float,
) -> list[_RawSegment]:
    """Two-state HMM segmentation for a single chromosome.

    Emissions are modelled as exponential densities on -log10(p):

    - **null** state: rate = *null_rate* (theoretical ``ln(10)``).
    - **associated** state: rate = *signal_rate* (smaller → stronger
      expected signals).

    Transition matrix is symmetric with off-diagonal *transition_prob*.
    """
    if not records:
        return []

    # Compute -log10(p) observations.
    p_values = np.array([float(r["p_value"]) for r in records], dtype=np.float64)
    p_values = np.clip(p_values, _P_FLOOR, 1.0)
    obs = -np.log10(p_values)

    # Log emission probabilities: log(rate) - rate * x  (exponential pdf).
    log_emit = np.empty((len(obs), 2), dtype=np.float64)
    log_emit[:, 0] = np.log(null_rate) - null_rate * obs
    log_emit[:, 1] = np.log(signal_rate) - signal_rate * obs

    # Transition matrix (log scale).
    log_trans = np.array([
        [np.log(1.0 - transition_prob), np.log(transition_prob)],
        [np.log(transition_prob), np.log(1.0 - transition_prob)],
    ])

    # Prior.
    log_prior = np.array([np.log(1.0 - prior_assoc), np.log(prior_assoc)])

    states = _viterbi_decode(log_emit, log_prior, log_trans)

    mask = [bool(s == 1) for s in states]
    return _collect_segments(records, mask)
