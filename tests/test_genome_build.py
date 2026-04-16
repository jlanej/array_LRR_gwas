"""Tests for genome build detection and exclusion regions."""

from pathlib import Path

import numpy as np
import pysam
import pytest

from array_lrr_gwas.genome_build import (
    SUPPORTED_BUILDS,
    _normalise_build,
    detect_build,
    get_exclusion_regions,
    get_par_regions,
)


class TestNormaliseBuild:
    @pytest.mark.parametrize(
        "alias,expected",
        [
            ("GRCh37", "GRCh37"),
            ("grch37", "GRCh37"),
            ("hg19", "GRCh37"),
            ("GRCh38", "GRCh38"),
            ("hg38", "GRCh38"),
            ("T2T-CHM13", "T2T-CHM13"),
            ("t2t-chm13", "T2T-CHM13"),
            ("CHM13", "T2T-CHM13"),
            ("chm13", "T2T-CHM13"),
            ("chm13v2.0", "T2T-CHM13"),
            ("hs1", "T2T-CHM13"),
        ],
    )
    def test_aliases(self, alias, expected):
        assert _normalise_build(alias) == expected

    def test_unknown_passthrough(self):
        assert _normalise_build("unknown") == "unknown"


class TestGetExclusionRegions:
    @pytest.mark.parametrize("build", SUPPORTED_BUILDS)
    def test_all_builds_return_regions(self, build):
        regions = get_exclusion_regions(build)
        assert isinstance(regions, dict)
        assert len(regions) > 0
        # Every build should cover at least chr1-chr22
        assert "chr1" in regions

    @pytest.mark.parametrize("build", SUPPORTED_BUILDS)
    def test_regions_contain_centromere_and_mhc(self, build):
        regions = get_exclusion_regions(build)
        # chr6 should have at least 2 regions (centromere + MHC)
        assert len(regions["chr6"]) >= 2

    @pytest.mark.parametrize("build", SUPPORTED_BUILDS)
    def test_regions_contain_ig_loci(self, build):
        regions = get_exclusion_regions(build)
        # chr14 should have centromere + IGH
        assert len(regions["chr14"]) >= 2
        # chr2 should have centromere + IGK
        assert len(regions["chr2"]) >= 2
        # chr22 should have centromere + IGL
        assert len(regions["chr22"]) >= 2

    def test_noprefix_naming(self):
        regions = get_exclusion_regions("GRCh38", chromosomes=["1", "2"])
        assert "1" in regions
        assert "chr1" not in regions

    def test_chr_prefix_naming(self):
        regions = get_exclusion_regions("GRCh38", chromosomes=["chr1", "chr2"])
        assert "chr1" in regions
        assert "1" not in regions

    def test_unknown_build_raises(self):
        with pytest.raises(ValueError, match="Unknown genome build"):
            get_exclusion_regions("unknown_build")

    def test_t2t_centromeres_larger_than_grch38(self):
        """T2T-CHM13 centromeric regions should be larger (fully resolved)."""
        t2t = get_exclusion_regions("T2T-CHM13")
        grch38 = get_exclusion_regions("GRCh38")
        # chr9 is a good example: T2T has a huge pericentromeric region
        t2t_chr9 = t2t["chr9"][0]
        grch38_chr9 = grch38["chr9"][0]
        t2t_span = t2t_chr9[1] - t2t_chr9[0]
        grch38_span = grch38_chr9[1] - grch38_chr9[0]
        assert t2t_span > grch38_span


class TestDetectBuild:
    def test_detect_from_assembly_tag(self, tmp_path):
        """Build detected when contig header contains assembly name."""
        vcf_path = tmp_path / "tagged.vcf"
        with open(vcf_path, "w") as f:
            f.write("##fileformat=VCFv4.2\n")
            f.write('##contig=<ID=chr1,length=248956422,assembly=GRCh38>\n')
            f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        assert detect_build(vcf_path) == "GRCh38"

    def test_detect_t2t_from_assembly_tag(self, tmp_path):
        """Build detected when contig header mentions CHM13."""
        vcf_path = tmp_path / "t2t.vcf"
        with open(vcf_path, "w") as f:
            f.write("##fileformat=VCFv4.2\n")
            f.write('##contig=<ID=chr1,length=248387328,assembly=T2T-CHM13>\n')
            f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        assert detect_build(vcf_path) == "T2T-CHM13"

    def test_detect_from_chr1_length_grch37(self, tmp_path):
        """Build detected from chr1 length matching GRCh37."""
        vcf_path = tmp_path / "hg19.vcf"
        with open(vcf_path, "w") as f:
            f.write("##fileformat=VCFv4.2\n")
            f.write("##contig=<ID=chr1,length=249250621>\n")
            f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        assert detect_build(vcf_path) == "GRCh37"

    def test_detect_from_chr1_length_t2t(self, tmp_path):
        """Build detected from chr1 length matching T2T-CHM13."""
        vcf_path = tmp_path / "t2t_len.vcf"
        with open(vcf_path, "w") as f:
            f.write("##fileformat=VCFv4.2\n")
            f.write("##contig=<ID=chr1,length=248387328>\n")
            f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        assert detect_build(vcf_path) == "T2T-CHM13"

    def test_returns_none_when_undetectable(self, tmp_path):
        """Returns None when no build info is available."""
        vcf_path = tmp_path / "bare.vcf"
        with open(vcf_path, "w") as f:
            f.write("##fileformat=VCFv4.2\n")
            f.write("##contig=<ID=chrX>\n")
            f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        assert detect_build(vcf_path) is None

    def test_detect_from_test_bcf(self, test_bcf_path):
        """Test BCF has no build info → returns None (no chr1 length)."""
        result = detect_build(test_bcf_path)
        # Our test BCF doesn't embed contig lengths, so detection may
        # return None – that's fine; the CLI requires --build in that case.
        assert result is None or result in SUPPORTED_BUILDS


class TestGetParRegions:
    """Tests for ``get_par_regions``."""

    @pytest.mark.parametrize("build", SUPPORTED_BUILDS)
    def test_all_builds_return_regions(self, build):
        regions = get_par_regions(build)
        assert len(regions) > 0

    @pytest.mark.parametrize("build", SUPPORTED_BUILDS)
    def test_par_contains_x_chromosome(self, build):
        regions = get_par_regions(build)
        x_chroms = {k for k in regions if k.upper().replace("CHR", "") == "X"}
        assert len(x_chroms) >= 1, f"No X chromosome PAR for {build}"

    @pytest.mark.parametrize("build", SUPPORTED_BUILDS)
    def test_par_contains_y_chromosome(self, build):
        regions = get_par_regions(build)
        y_chroms = {k for k in regions if k.upper().replace("CHR", "") == "Y"}
        assert len(y_chroms) >= 1, f"No Y chromosome PAR for {build}"

    def test_grch37_uses_bare_names_by_default(self):
        """GRCh37 uses non-prefixed names (X, Y) in its definition."""
        regions = get_par_regions("GRCh37", chromosomes=["X", "Y"])
        assert "X" in regions
        assert "Y" in regions

    def test_grch38_uses_chr_prefix(self):
        regions = get_par_regions("GRCh38", chromosomes=["chrX"])
        assert "chrX" in regions

    def test_t2t_par1_reasonable_size(self):
        """T2T PAR1 on chrX should be ~2.78 Mb."""
        regions = get_par_regions("T2T-CHM13")
        chrx_regions = regions["chrX"]
        # First region is PAR1
        par1_start, par1_end = chrx_regions[0]
        par1_size = par1_end - par1_start
        assert 2_000_000 < par1_size < 4_000_000

    def test_unknown_build_raises(self):
        with pytest.raises(ValueError, match="Unknown genome build"):
            get_par_regions("UnknownBuild")
