"""Tests for marker subsetting."""

import numpy as np
import pytest

from array_lrr_gwas.subsetting import (
    autosome_mask,
    call_rate_mask,
    complexity_mask,
    subset_markers,
    variance_mask,
)


class TestCallRateMask:
    def test_all_present(self):
        lrr = np.ones((10, 5))
        mask = call_rate_mask(lrr, min_call_rate=0.95)
        assert mask.all()

    def test_high_missingness_excluded(self):
        lrr = np.ones((10, 20))
        lrr[0, :5] = np.nan  # 75% call rate
        mask = call_rate_mask(lrr, min_call_rate=0.80)
        assert not mask[0]
        assert mask[1:].all()

    def test_boundary(self):
        lrr = np.ones((1, 100))
        lrr[0, :5] = np.nan  # exactly 95%
        mask = call_rate_mask(lrr, min_call_rate=0.95)
        assert mask[0]

    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="2-D"):
            call_rate_mask(np.ones(5))


class TestVarianceMask:
    def test_constant_excluded(self):
        lrr = np.zeros((10, 5))
        mask = variance_mask(lrr, min_var=0.001)
        assert not mask.any()

    def test_high_variance_excluded(self):
        lrr = np.random.default_rng(0).normal(0, 10, (10, 50))
        mask = variance_mask(lrr, min_var=0.001, max_var=1.0)
        # All rows have variance >> 1
        assert not mask.any()

    def test_normal_passes(self):
        rng = np.random.default_rng(0)
        lrr = rng.normal(0, 0.1, (10, 50))
        mask = variance_mask(lrr, min_var=0.001, max_var=1.0)
        assert mask.all()

    def test_no_upper_bound(self):
        rng = np.random.default_rng(0)
        lrr = rng.normal(0, 10, (5, 50))
        mask = variance_mask(lrr, min_var=0.001, max_var=None)
        assert mask.all()


class TestComplexityMask:
    def test_no_exclusions(self):
        pos = np.array([100, 200, 300])
        chroms = ["chr1", "chr1", "chr2"]
        mask = complexity_mask(pos, chroms, exclude_regions=None)
        assert mask.all()

    def test_region_excluded(self):
        pos = np.array([100, 200, 300, 400])
        chroms = ["chr1", "chr1", "chr1", "chr2"]
        exclude = {"chr1": [(150, 250)]}
        mask = complexity_mask(pos, chroms, exclude_regions=exclude)
        assert mask[0]  # 100, outside
        assert not mask[1]  # 200, inside
        assert mask[2]  # 300, outside
        assert mask[3]  # chr2, not excluded


class TestAutosomeMask:
    def test_keeps_autosomes(self):
        chroms = ["chr1", "chr2", "chr22"]
        mask = autosome_mask(chroms)
        assert mask.all()

    def test_excludes_sex_and_mt(self):
        chroms = ["chrX", "chrY", "chrM", "chrMT"]
        mask = autosome_mask(chroms)
        assert not mask.any()

    def test_bare_names(self):
        chroms = ["1", "X", "Y", "MT", "22"]
        mask = autosome_mask(chroms)
        expected = [True, False, False, False, True]
        np.testing.assert_array_equal(mask, expected)

    def test_mixed(self):
        chroms = ["chr1", "chrX", "chr2", "chrY"]
        mask = autosome_mask(chroms)
        expected = [True, False, True, False]
        np.testing.assert_array_equal(mask, expected)


class TestSubsetMarkers:
    def test_combined_filters(self, synthetic_lrr):
        mask = subset_markers(
            synthetic_lrr,
            min_call_rate=0.60,
            min_var=0.0001,
        )
        # Marker 0 has >50% missing -> may or may not pass at 0.60
        # Marker 1 is constant -> should fail variance filter
        assert not mask[1], "Constant marker should be excluded"

    def test_returns_bool_array(self, synthetic_lrr):
        mask = subset_markers(synthetic_lrr, min_call_rate=0.5, min_var=0.0)
        assert mask.dtype == bool
        assert mask.shape == (synthetic_lrr.shape[0],)

    def test_autosome_filter_excludes_sex_chroms(self):
        rng = np.random.default_rng(0)
        lrr = rng.normal(0, 0.1, (4, 20))
        chroms = ["chr1", "chrX", "chr2", "chrY"]
        mask = subset_markers(
            lrr,
            chromosomes=chroms,
            min_call_rate=0.0,
            min_var=0.0,
            autosomes_only=True,
        )
        assert mask[0]  # chr1 kept
        assert not mask[1]  # chrX excluded
        assert mask[2]  # chr2 kept
        assert not mask[3]  # chrY excluded

    def test_autosome_filter_disabled(self):
        rng = np.random.default_rng(0)
        lrr = rng.normal(0, 0.1, (3, 20))
        chroms = ["chr1", "chrX", "chr2"]
        mask = subset_markers(
            lrr,
            chromosomes=chroms,
            min_call_rate=0.0,
            min_var=0.0,
            autosomes_only=False,
        )
        assert mask.all()
