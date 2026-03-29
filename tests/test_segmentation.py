"""Tests for the post-GWAS segmentation module."""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

from array_lrr_gwas.segmentation import (
    SegmentationResult,
    _collect_segments,
    _hmm_segment,
    _threshold_segment,
    _viterbi_decode,
    read_association_tsv,
    segment_associations,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_records(
    n: int,
    *,
    chrom: str = "chr1",
    start_pos: int = 1000,
    step: int = 100,
    p_value: float = 0.5,
    beta: float = 0.0,
    stat: float = 0.0,
    method: str = "ols",
) -> list[dict[str, object]]:
    """Factory for minimal association records."""
    return [
        {
            "chrom": chrom,
            "pos": start_pos + i * step,
            "variant_id": f"snp{i}",
            "beta": beta,
            "se": 0.1,
            "stat": stat,
            "p_value": p_value,
            "n_samples": 100,
            "method": method,
        }
        for i in range(n)
    ]


def _inject_signal(
    records: list[dict[str, object]],
    indices: list[int],
    p_value: float = 1e-12,
    beta: float = 0.5,
    stat: float = 7.0,
) -> None:
    """Overwrite selected records with a strong signal."""
    for idx in indices:
        records[idx] = {**records[idx], "p_value": p_value, "beta": beta, "stat": stat}


# ---------------------------------------------------------------------------
# Viterbi decoding
# ---------------------------------------------------------------------------


class TestViterbiDecode:
    def test_empty_input(self) -> None:
        log_emit = np.empty((0, 2))
        log_prior = np.log([0.5, 0.5])
        log_trans = np.log([[0.9, 0.1], [0.1, 0.9]])
        states = _viterbi_decode(log_emit, log_prior, log_trans)
        assert len(states) == 0

    def test_single_observation(self) -> None:
        # High emission for state 1 → should decode to state 1.
        log_emit = np.array([[np.log(0.01), np.log(0.99)]])
        log_prior = np.log([0.5, 0.5])
        log_trans = np.log([[0.9, 0.1], [0.1, 0.9]])
        states = _viterbi_decode(log_emit, log_prior, log_trans)
        assert states[0] == 1

    def test_strong_state_block(self) -> None:
        """Block of high-emission observations → contiguous state 1."""
        T = 20
        log_emit = np.zeros((T, 2))
        # First 10: strong null, last 10: strong signal.
        log_emit[:10, 0] = np.log(0.99)
        log_emit[:10, 1] = np.log(0.01)
        log_emit[10:, 0] = np.log(0.01)
        log_emit[10:, 1] = np.log(0.99)

        log_prior = np.log([0.9, 0.1])
        log_trans = np.log([[0.95, 0.05], [0.05, 0.95]])
        states = _viterbi_decode(log_emit, log_prior, log_trans)

        assert all(s == 0 for s in states[:10])
        assert all(s == 1 for s in states[10:])


# ---------------------------------------------------------------------------
# _collect_segments
# ---------------------------------------------------------------------------


class TestCollectSegments:
    def test_no_flagged_markers(self) -> None:
        recs = _make_records(5)
        mask = [False] * 5
        assert _collect_segments(recs, mask) == []

    def test_single_marker_segment(self) -> None:
        recs = _make_records(5)
        mask = [False, False, True, False, False]
        segs = _collect_segments(recs, mask)
        assert len(segs) == 1
        assert segs[0].n_markers == 1
        # BED: 0-based start = pos - 1
        assert segs[0].start == recs[2]["pos"] - 1
        assert segs[0].end == recs[2]["pos"]

    def test_contiguous_run(self) -> None:
        recs = _make_records(5)
        mask = [False, True, True, True, False]
        segs = _collect_segments(recs, mask)
        assert len(segs) == 1
        assert segs[0].n_markers == 3
        assert segs[0].start == recs[1]["pos"] - 1
        assert segs[0].end == recs[3]["pos"]

    def test_multiple_runs(self) -> None:
        recs = _make_records(7)
        mask = [True, True, False, False, True, True, True]
        segs = _collect_segments(recs, mask)
        assert len(segs) == 2
        assert segs[0].n_markers == 2
        assert segs[1].n_markers == 3

    def test_statistics_computed(self) -> None:
        recs = _make_records(3, p_value=0.01, beta=0.2, stat=3.0)
        recs[1]["p_value"] = 1e-10
        recs[1]["beta"] = 0.5
        recs[1]["stat"] = -6.0
        mask = [True, True, True]
        segs = _collect_segments(recs, mask)
        assert len(segs) == 1
        assert segs[0].min_p == pytest.approx(1e-10)
        assert segs[0].max_abs_stat == pytest.approx(6.0)
        expected_mean_beta = (0.2 + 0.5 + 0.2) / 3
        assert segs[0].mean_beta == pytest.approx(expected_mean_beta)


# ---------------------------------------------------------------------------
# Threshold strategy
# ---------------------------------------------------------------------------


class TestThresholdSegment:
    def test_empty_input(self) -> None:
        assert _threshold_segment([], 5e-8, 1_000_000) == []

    def test_no_significant_markers(self) -> None:
        recs = _make_records(10, p_value=0.5)
        segs = _threshold_segment(recs, 5e-8, 1_000_000)
        assert segs == []

    def test_all_significant(self) -> None:
        recs = _make_records(10, p_value=1e-10)
        segs = _threshold_segment(recs, 5e-8, 1_000_000)
        assert len(segs) == 1
        assert segs[0].n_markers == 10

    def test_merge_nearby(self) -> None:
        """Two flagged blocks separated by < max_gap should merge."""
        recs = _make_records(10, step=100, p_value=0.5)
        _inject_signal(recs, [1, 2])
        _inject_signal(recs, [5, 6])
        # Gap between marker 2 (pos 1200) and marker 5 (pos 1500) is 300 bp.
        segs = _threshold_segment(recs, 5e-8, 1_000_000)
        assert len(segs) == 1

    def test_no_merge_beyond_gap(self) -> None:
        """Two flagged blocks separated by > max_gap stay separate."""
        recs = _make_records(10, step=100, p_value=0.5)
        _inject_signal(recs, [1, 2])
        _inject_signal(recs, [5, 6])
        # Gap ~300 bp, set max_gap to 100 to prevent merging.
        segs = _threshold_segment(recs, 5e-8, 100)
        assert len(segs) == 2


# ---------------------------------------------------------------------------
# HMM strategy
# ---------------------------------------------------------------------------


class TestHmmSegment:
    def test_empty_input(self) -> None:
        assert _hmm_segment([], 2.303, 0.1, 1e-3, 1e-4) == []

    def test_no_signal(self) -> None:
        """All null p-values → should produce zero or very few segments."""
        rng = np.random.default_rng(42)
        recs = _make_records(100, p_value=0.5)
        # Random null p-values.
        for r in recs:
            r["p_value"] = float(rng.uniform(0.01, 1.0))
        segs = _hmm_segment(recs, 2.303, 0.1, 1e-3, 1e-4)
        # Under null, HMM should rarely call associated state.
        total_markers = sum(s.n_markers for s in segs)
        assert total_markers < 10  # tolerant bound

    def test_strong_signal_detected(self) -> None:
        """Clear signal block → HMM should identify it."""
        recs = _make_records(50, p_value=0.5, beta=0.0, stat=0.0)
        _inject_signal(recs, list(range(20, 30)), p_value=1e-15, stat=8.0)
        segs = _hmm_segment(recs, 2.303, 0.1, 1e-3, 1e-4)
        assert len(segs) >= 1
        # The signal region should be captured.
        signal_markers = sum(s.n_markers for s in segs)
        assert signal_markers >= 5  # at least half of signal block

    def test_handles_p_zero(self) -> None:
        """p-value of exactly 0.0 should not cause errors."""
        recs = _make_records(5, p_value=0.5)
        recs[2]["p_value"] = 0.0
        # Should not raise.
        _hmm_segment(recs, 2.303, 0.1, 1e-3, 1e-4)


# ---------------------------------------------------------------------------
# Main entry point: segment_associations
# ---------------------------------------------------------------------------


class TestSegmentAssociations:
    def test_invalid_strategy(self) -> None:
        recs = _make_records(5)
        with pytest.raises(ValueError, match="Unknown strategy"):
            segment_associations(recs, strategy="invalid")

    def test_empty_input(self) -> None:
        result = segment_associations([], strategy="threshold")
        assert len(result.chrom) == 0
        assert result.strategy == "threshold"

    def test_hmm_strategy_recorded(self) -> None:
        recs = _make_records(10, p_value=1e-10)
        result = segment_associations(recs, strategy="hmm")
        assert result.strategy == "hmm"
        assert "null_rate" in result.parameters

    def test_threshold_strategy_recorded(self) -> None:
        recs = _make_records(10, p_value=1e-10)
        result = segment_associations(recs, strategy="threshold")
        assert result.strategy == "threshold"
        assert "p_threshold" in result.parameters

    def test_min_markers_filter(self) -> None:
        """Segments with fewer than min_markers should be dropped."""
        recs = _make_records(10, p_value=0.5)
        # Single significant marker.
        _inject_signal(recs, [5])
        result = segment_associations(
            recs, strategy="threshold", min_markers=3,
        )
        assert len(result.chrom) == 0  # single marker filtered out

    def test_multi_chromosome(self) -> None:
        """Segments on different chromosomes are independent."""
        recs_chr1 = _make_records(10, chrom="chr1", p_value=1e-10)
        recs_chr2 = _make_records(10, chrom="chr2", start_pos=5000, p_value=1e-10)
        recs = recs_chr1 + recs_chr2
        result = segment_associations(recs, strategy="threshold")
        assert len(result.chrom) >= 2
        assert "chr1" in result.chrom
        assert "chr2" in result.chrom

    def test_region_names_sequential(self) -> None:
        recs = _make_records(10, p_value=1e-10, step=200)
        result = segment_associations(recs, strategy="threshold")
        for i, name in enumerate(result.name, start=1):
            assert name == f"region_{i}"


# ---------------------------------------------------------------------------
# SegmentationResult output
# ---------------------------------------------------------------------------


class TestSegmentationResult:
    def test_to_records_schema(self) -> None:
        """Every record has all BED+ fields."""
        recs = _make_records(10, p_value=1e-10)
        result = segment_associations(recs, strategy="threshold")
        required_keys = {
            "chrom", "start", "end", "name", "n_markers",
            "min_p", "mean_beta", "max_abs_stat", "method",
        }
        for rec in result.to_records():
            assert required_keys.issubset(rec.keys())

    def test_bed_coordinates(self) -> None:
        """Start is 0-based, end is exclusive (BED convention)."""
        recs = _make_records(5, p_value=1e-10, start_pos=1000, step=100)
        result = segment_associations(recs, strategy="threshold")
        assert len(result.start) >= 1
        # start should be 0-based (pos-1 of first marker).
        assert result.start[0] == 999
        # end should be pos of last marker (0-based exclusive).
        assert result.end[0] == 1400

    def test_write_bed(self, tmp_path: Path) -> None:
        recs = _make_records(10, p_value=1e-10)
        result = segment_associations(recs, strategy="threshold")
        bed_path = tmp_path / "output.bed"
        result.write_bed(bed_path)

        lines = bed_path.read_text().strip().split("\n")
        # First line is comment header.
        assert lines[0].startswith("#")
        header_cols = lines[0].lstrip("#").split("\t")
        assert "chrom" in header_cols
        assert "start" in header_cols
        # Data lines present.
        assert len(lines) > 1

    def test_write_bed_empty(self, tmp_path: Path) -> None:
        """Empty result writes only the header."""
        result = segment_associations([], strategy="threshold")
        bed_path = tmp_path / "empty.bed"
        result.write_bed(bed_path)
        lines = bed_path.read_text().strip().split("\n")
        assert lines[0].startswith("#")
        assert len(lines) == 1

    def test_numeric_types(self) -> None:
        """Numeric fields in records should be plain Python types."""
        recs = _make_records(5, p_value=1e-10)
        result = segment_associations(recs, strategy="threshold")
        for rec in result.to_records():
            assert isinstance(rec["start"], int)
            assert isinstance(rec["end"], int)
            assert isinstance(rec["n_markers"], int)
            assert isinstance(rec["min_p"], float)
            assert isinstance(rec["mean_beta"], float)
            assert isinstance(rec["max_abs_stat"], float)


# ---------------------------------------------------------------------------
# read_association_tsv
# ---------------------------------------------------------------------------


class TestReadAssociationTsv:
    def test_round_trip(self, tmp_path: Path) -> None:
        """Write a TSV, read it back, verify types and values."""
        tsv_path = tmp_path / "results.tsv"
        tsv_path.write_text(textwrap.dedent("""\
            chrom\tpos\tvariant_id\tbeta\tse\tstat\tp_value\tn_samples\tmethod
            chr1\t1000\tv0\t0.25\t0.05\t5.0\t1e-06\t200\tols
            chr1\t2000\tv1\t-0.10\t0.08\t-1.25\t0.21\t200\tols
        """))
        records = read_association_tsv(tsv_path)
        assert len(records) == 2
        assert records[0]["chrom"] == "chr1"
        assert records[0]["pos"] == 1000
        assert records[0]["p_value"] == pytest.approx(1e-6)
        assert isinstance(records[1]["n_samples"], int)
