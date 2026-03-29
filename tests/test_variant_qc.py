"""Tests for the variant_qc module."""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

from array_lrr_gwas.variant_qc import (
    VariantQCRecord,
    read_collated_variant_qc,
    variant_qc_mask,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_tsv(tmp_path: Path, content: str, filename: str = "qc.tsv") -> Path:
    """Write *content* to a TSV file under *tmp_path* and return the path."""
    p = tmp_path / filename
    p.write_text(textwrap.dedent(content))
    return p


_HEADER = (
    "variant_id\tall_ancestries_call_rate_pass\t"
    "all_ancestries_hwe_pass\tall_ancestries_maf_pass\n"
)


def _make_qc_tsv(tmp_path: Path, rows: list[str], filename: str = "qc.tsv") -> Path:
    """Create a TSV with the standard header and *rows* lines."""
    text = _HEADER + "\n".join(rows) + "\n"
    return _write_tsv(tmp_path, text, filename)


# ---------------------------------------------------------------------------
# read_collated_variant_qc
# ---------------------------------------------------------------------------


class TestReadCollatedVariantQC:
    """Tests for :func:`read_collated_variant_qc`."""

    def test_basic_parse(self, tmp_path: Path) -> None:
        tsv = _make_qc_tsv(tmp_path, [
            "v1\tTrue\tTrue\tTrue",
            "v2\tFalse\tTrue\tFalse",
            "v3\tTrue\tFalse\tTrue",
        ])
        data = read_collated_variant_qc(tsv)
        assert len(data) == 3
        assert data["v1"].call_rate_pass is True
        assert data["v2"].call_rate_pass is False
        assert data["v3"].hwe_pass is False
        assert data["v2"].maf_pass is False
        assert data["v3"].maf_pass is True

    def test_boolean_literals(self, tmp_path: Path) -> None:
        """Accept 'True', '1', 'yes' (case-insensitive) as truthy."""
        tsv = _make_qc_tsv(tmp_path, [
            "a\tTrue\t1\tyes",
            "b\tFALSE\t0\tno",
        ])
        data = read_collated_variant_qc(tsv)
        assert data["a"].call_rate_pass is True
        assert data["a"].hwe_pass is True
        assert data["a"].maf_pass is True
        assert data["b"].call_rate_pass is False
        assert data["b"].hwe_pass is False
        assert data["b"].maf_pass is False

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            read_collated_variant_qc(tmp_path / "nonexistent.tsv")

    def test_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.tsv"
        p.write_text("")
        with pytest.raises(ValueError, match="empty or has no header"):
            read_collated_variant_qc(p)

    def test_header_only_no_data(self, tmp_path: Path) -> None:
        p = _write_tsv(tmp_path, _HEADER)
        with pytest.raises(ValueError, match="no data rows"):
            read_collated_variant_qc(p)

    def test_missing_columns(self, tmp_path: Path) -> None:
        p = _write_tsv(tmp_path, "variant_id\tall_ancestries_call_rate_pass\nv1\tTrue\n")
        with pytest.raises(ValueError, match="missing required columns"):
            read_collated_variant_qc(p)

    def test_duplicate_variant_warns(self, tmp_path: Path) -> None:
        tsv = _make_qc_tsv(tmp_path, [
            "v1\tTrue\tTrue\tTrue",
            "v1\tFalse\tFalse\tFalse",
            "v2\tTrue\tTrue\tTrue",
        ])
        with pytest.warns(UserWarning, match="duplicate variant_id"):
            data = read_collated_variant_qc(tsv)
        # First occurrence kept.
        assert data["v1"].call_rate_pass is True
        assert len(data) == 2

    def test_extra_columns_ignored(self, tmp_path: Path) -> None:
        """Extra columns beyond the required four should be silently ignored."""
        header = (
            "variant_id\tall_ancestries_call_rate_pass\t"
            "all_ancestries_hwe_pass\tall_ancestries_maf_pass\textra_col\n"
        )
        p = _write_tsv(tmp_path, header + "v1\tTrue\tTrue\tTrue\t42\n")
        data = read_collated_variant_qc(p)
        assert len(data) == 1
        assert data["v1"].call_rate_pass is True


# ---------------------------------------------------------------------------
# variant_qc_mask — exact match
# ---------------------------------------------------------------------------


class TestVariantQCMaskExactMatch:
    """Mask with variant_ids perfectly matching QC data."""

    @pytest.fixture()
    def qc_data(self, tmp_path: Path) -> dict[str, VariantQCRecord]:
        tsv = _make_qc_tsv(tmp_path, [
            "v1\tTrue\tTrue\tTrue",
            "v2\tTrue\tTrue\tFalse",
            "v3\tFalse\tTrue\tTrue",
            "v4\tTrue\tFalse\tTrue",
        ])
        return read_collated_variant_qc(tsv)

    def test_default_filters(self, qc_data: dict) -> None:
        """Default: require_call_rate + require_hwe, not MAF."""
        ids = ["v1", "v2", "v3", "v4"]
        mask = variant_qc_mask(ids, qc_data)
        # v1: pass, v2: pass (MAF not required), v3: fail call_rate, v4: fail HWE
        np.testing.assert_array_equal(mask, [True, True, False, False])

    def test_maf_required(self, qc_data: dict) -> None:
        ids = ["v1", "v2", "v3", "v4"]
        mask = variant_qc_mask(ids, qc_data, require_maf=True)
        # v1: pass, v2: fail maf, v3: fail call_rate, v4: fail HWE
        np.testing.assert_array_equal(mask, [True, False, False, False])

    def test_no_filters(self, qc_data: dict) -> None:
        ids = ["v1", "v2", "v3", "v4"]
        mask = variant_qc_mask(
            ids, qc_data,
            require_call_rate=False, require_hwe=False, require_maf=False,
        )
        np.testing.assert_array_equal(mask, [True, True, True, True])


# ---------------------------------------------------------------------------
# variant_qc_mask — reordered VCF
# ---------------------------------------------------------------------------


class TestVariantQCMaskReordered:
    """Mask correctly aligns when variant_ids are in a different order."""

    def test_reordered(self, tmp_path: Path) -> None:
        tsv = _make_qc_tsv(tmp_path, [
            "v1\tTrue\tTrue\tTrue",
            "v2\tFalse\tTrue\tTrue",
            "v3\tTrue\tTrue\tTrue",
        ])
        qc = read_collated_variant_qc(tsv)
        # Request in reverse order.
        mask = variant_qc_mask(["v3", "v1", "v2"], qc)
        # v3: pass, v1: pass, v2: fail call_rate
        np.testing.assert_array_equal(mask, [True, True, False])


# ---------------------------------------------------------------------------
# variant_qc_mask — missing variants
# ---------------------------------------------------------------------------


class TestVariantQCMaskMissing:
    """Variants with no QC record should be excluded (conservative)."""

    def test_missing_variants_excluded_and_warned(self, tmp_path: Path) -> None:
        tsv = _make_qc_tsv(tmp_path, [
            "v1\tTrue\tTrue\tTrue",
        ])
        qc = read_collated_variant_qc(tsv)
        with pytest.warns(UserWarning, match="no matching QC record"):
            mask = variant_qc_mask(["v1", "v_unknown"], qc)
        np.testing.assert_array_equal(mask, [True, False])


# ---------------------------------------------------------------------------
# variant_qc_mask — extra QC records
# ---------------------------------------------------------------------------


class TestVariantQCMaskExtra:
    """Extra QC records not in variant_ids should be silently ignored."""

    def test_extra_qc_records(self, tmp_path: Path) -> None:
        tsv = _make_qc_tsv(tmp_path, [
            "v1\tTrue\tTrue\tTrue",
            "v2\tTrue\tTrue\tTrue",
            "v_extra\tTrue\tTrue\tTrue",
        ])
        qc = read_collated_variant_qc(tsv)
        mask = variant_qc_mask(["v1", "v2"], qc)
        np.testing.assert_array_equal(mask, [True, True])


# ---------------------------------------------------------------------------
# variant_qc_mask — fallback (no QC data)
# ---------------------------------------------------------------------------


class TestVariantQCMaskFallback:
    """When qc_data is None a warning is emitted and an all-True mask returned."""

    def test_none_qc_data(self) -> None:
        with pytest.warns(UserWarning, match="No upstream variant QC data"):
            mask = variant_qc_mask(["v1", "v2", "v3"], None)
        np.testing.assert_array_equal(mask, [True, True, True])

    def test_fallback_shape(self) -> None:
        with pytest.warns(UserWarning):
            mask = variant_qc_mask(["a", "b", "c", "d", "e"], None)
        assert mask.shape == (5,)
        assert mask.dtype == np.bool_


# ---------------------------------------------------------------------------
# Multi-ancestry edge cases
# ---------------------------------------------------------------------------


class TestMultiAncestryEdgeCases:
    """Ensure correct handling of cross-ancestry flag combinations."""

    def test_all_fail(self, tmp_path: Path) -> None:
        tsv = _make_qc_tsv(tmp_path, [
            "v1\tFalse\tFalse\tFalse",
        ])
        qc = read_collated_variant_qc(tsv)
        mask = variant_qc_mask(["v1"], qc, require_maf=True)
        np.testing.assert_array_equal(mask, [False])

    def test_all_pass(self, tmp_path: Path) -> None:
        tsv = _make_qc_tsv(tmp_path, [
            "v1\tTrue\tTrue\tTrue",
        ])
        qc = read_collated_variant_qc(tsv)
        mask = variant_qc_mask(["v1"], qc, require_maf=True)
        np.testing.assert_array_equal(mask, [True])

    def test_only_maf_fails(self, tmp_path: Path) -> None:
        """MAF fail should not affect mask when require_maf=False."""
        tsv = _make_qc_tsv(tmp_path, [
            "v1\tTrue\tTrue\tFalse",
        ])
        qc = read_collated_variant_qc(tsv)
        # MAF not required (RSVD default) → variant retained.
        mask_no_maf = variant_qc_mask(["v1"], qc, require_maf=False)
        np.testing.assert_array_equal(mask_no_maf, [True])
        # MAF required (GRM) → variant excluded.
        mask_maf = variant_qc_mask(["v1"], qc, require_maf=True)
        np.testing.assert_array_equal(mask_maf, [False])

    def test_large_mixed(self, tmp_path: Path) -> None:
        """Larger TSV with mixed pass/fail per ancestry flag."""
        rows = []
        expected = []
        for i in range(100):
            cr = "True" if i % 2 == 0 else "False"
            hwe = "True" if i % 3 != 0 else "False"
            maf = "True"
            rows.append(f"v{i}\t{cr}\t{hwe}\t{maf}")
            expected.append(i % 2 == 0 and i % 3 != 0)
        tsv = _make_qc_tsv(tmp_path, rows)
        qc = read_collated_variant_qc(tsv)
        ids = [f"v{i}" for i in range(100)]
        mask = variant_qc_mask(ids, qc)
        np.testing.assert_array_equal(mask, expected)
