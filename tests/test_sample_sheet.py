"""Tests for sample sheet parsing (array_lrr_gwas.sample_sheet)."""

from __future__ import annotations

import numpy as np
import pytest

from array_lrr_gwas.sample_sheet import read_sample_sheet, align_samples


class TestReadSampleSheet:
    """Tests for ``read_sample_sheet``."""

    def test_basic_parsing(self, tmp_path) -> None:
        tsv = tmp_path / "sheet.tsv"
        tsv.write_text(
            "Sample_ID\tPC1\tPC2\tPC3\n"
            "S1\t0.1\t0.2\t0.3\n"
            "S2\t0.4\t0.5\t0.6\n"
        )
        ids, covs, names = read_sample_sheet(tsv)
        assert ids == ["S1", "S2"]
        assert names == ["PC1", "PC2", "PC3"]
        assert covs.shape == (2, 3)
        np.testing.assert_allclose(covs[0], [0.1, 0.2, 0.3])

    def test_n_pcs_limit(self, tmp_path) -> None:
        tsv = tmp_path / "sheet.tsv"
        cols = "\t".join(f"PC{i}" for i in range(1, 6))
        vals = "\t".join("1.0" for _ in range(5))
        tsv.write_text(f"Sample_ID\t{cols}\nS1\t{vals}\n")
        _, covs, names = read_sample_sheet(tsv, n_pcs=3)
        assert len(names) == 3
        assert covs.shape == (1, 3)
        assert names == ["PC1", "PC2", "PC3"]

    def test_extra_covariates(self, tmp_path) -> None:
        tsv = tmp_path / "sheet.tsv"
        tsv.write_text(
            "Sample_ID\tPC1\tpredicted_sex\n"
            "S1\t0.1\t1\n"
            "S2\t0.2\t0\n"
        )
        _, covs, names = read_sample_sheet(
            tsv, extra_covariates=["predicted_sex"],
        )
        assert names == ["PC1", "predicted_sex"]
        assert covs.shape == (2, 2)

    def test_missing_values(self, tmp_path) -> None:
        tsv = tmp_path / "sheet.tsv"
        tsv.write_text(
            "Sample_ID\tPC1\tPC2\n"
            "S1\t0.1\tNA\n"
            "S2\t\t0.5\n"
        )
        _, covs, _ = read_sample_sheet(tsv)
        assert np.isnan(covs[0, 1])
        assert np.isnan(covs[1, 0])

    def test_empty_raises(self, tmp_path) -> None:
        tsv = tmp_path / "sheet.tsv"
        tsv.write_text("")
        with pytest.raises(ValueError, match="empty"):
            read_sample_sheet(tsv)

    def test_custom_sample_id_col(self, tmp_path) -> None:
        tsv = tmp_path / "sheet.tsv"
        tsv.write_text("IID\tPC1\nS1\t0.1\n")
        ids, _, _ = read_sample_sheet(tsv, sample_id_col="IID")
        assert ids == ["S1"]

    def test_pc_ordering(self, tmp_path) -> None:
        """PCs should be sorted numerically by index, not alphabetically."""
        tsv = tmp_path / "sheet.tsv"
        tsv.write_text("Sample_ID\tPC10\tPC2\tPC1\nS1\t10.0\t2.0\t1.0\n")
        _, covs, names = read_sample_sheet(tsv)
        assert names == ["PC1", "PC2", "PC10"]
        np.testing.assert_allclose(covs[0], [1.0, 2.0, 10.0])


class TestAlignSamples:
    """Tests for ``align_samples``."""

    def test_reorder(self) -> None:
        target = ["B", "A", "C"]
        sheet = ["A", "B", "C"]
        covs = np.array([[1.0], [2.0], [3.0]])
        aligned = align_samples(target, sheet, covs)
        np.testing.assert_allclose(aligned, [[2.0], [1.0], [3.0]])

    def test_missing_filled_with_nan(self) -> None:
        target = ["A", "B", "D"]
        sheet = ["A", "B"]
        covs = np.array([[1.0], [2.0]])
        aligned = align_samples(target, sheet, covs)
        assert aligned.shape == (3, 1)
        assert np.isnan(aligned[2, 0])
