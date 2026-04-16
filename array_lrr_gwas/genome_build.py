"""Genome build detection and default genomic exclusion regions.

Detects the reference genome build (GRCh37 / GRCh38 / T2T-CHM13) from
BCF/VCF contig metadata and provides curated exclusion regions for each
build.  Exclusion regions include centromeres, the MHC/HLA locus, and
immunoglobulin heavy- and light-chain loci – all of which are known to
confound array-based LRR analyses due to structural complexity, segmental
duplication, or copy-number polymorphism.

Also provides pseudoautosomal region (PAR1/PAR2) and X-transposed region
(XTR) coordinates for each build.  PAR/XTR regions on the X chromosome
recombine and segregate like autosomes; they must be excluded from
X-chromosome GRM (X-GRM) computation.

T2T-CHM13 (CHM13v2.0) is included because the fully resolved centromeric
satellite arrays are substantially larger than the gap-based coordinates
in GRCh37/38 and must be excluded from batch-effect estimation.

Sources
-------
- GRCh37 centromeres: UCSC Genome Browser hg19 gap track.
- GRCh38 centromeres: GRCh38 centromere annotations.
- T2T-CHM13 centromeres: CHM13v2.0 cenSat annotation track
  (Altemose et al. 2022, *Science*).
- MHC/HLA: classical MHC region boundaries (chr6).
- Immunoglobulin loci: IGH (chr14), IGK (chr2), IGL (chr22).
- PAR/XTR regions: derived from illumina_idat_processing utils.sh
  (Laurie et al. 2012, *Genet Epidemiol* 36:384–91).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pysam

logger = logging.getLogger(__name__)

# ── Known chromosome 1 lengths for build detection ────────────────────────
_CHR1_LENGTHS: dict[str, int] = {
    "T2T-CHM13": 248_387_328,
    "GRCh38": 248_956_422,
    "GRCh37": 249_250_621,
}

# Tolerance for length matching (handles minor patch differences)
_LENGTH_TOLERANCE = 100_000

# ── Exclusion region catalogues ───────────────────────────────────────────
# Each catalogue is a dict mapping chromosome → list of (start, end) tuples.
# Coordinates are 1-based, inclusive – consistent with VCF POS.

# ---------- GRCh37 (hg19) ----------
_GRCH37_REGIONS: dict[str, list[tuple[int, int]]] = {
    "chr1":  [(121_535_434, 124_535_434)],
    "chr2":  [(92_326_171, 95_326_171)],
    "chr3":  [(90_504_854, 93_504_854)],
    "chr4":  [(49_660_117, 52_660_117)],
    "chr5":  [(46_405_641, 49_405_641)],
    "chr6":  [(58_830_166, 61_830_166),
              # MHC / HLA region
              (28_477_797, 33_448_354)],
    "chr7":  [(58_054_331, 61_054_331)],
    "chr8":  [(43_838_887, 46_838_887)],
    "chr9":  [(47_367_679, 50_367_679)],
    "chr10": [(39_254_935, 42_254_935)],
    "chr11": [(51_644_566, 54_644_566)],
    "chr12": [(34_856_694, 37_856_694)],
    "chr13": [(16_000_000, 19_000_000)],
    "chr14": [(16_000_000, 19_000_000),
              # IGH – immunoglobulin heavy chain
              (106_032_614, 107_288_051)],
    "chr15": [(17_000_000, 20_000_000)],
    "chr16": [(35_335_801, 38_335_801)],
    "chr17": [(22_263_006, 25_263_006)],
    "chr18": [(15_460_898, 18_460_898)],
    "chr19": [(24_681_782, 27_681_782)],
    "chr20": [(26_369_569, 29_369_569)],
    "chr21": [(11_288_129, 14_288_129)],
    "chr22": [(13_000_000, 16_000_000),
              # IGL – immunoglobulin lambda
              (22_380_474, 23_265_085)],
}
# IGK – immunoglobulin kappa (chr2, separate from centromere)
_GRCH37_REGIONS["chr2"].append((89_156_874, 89_630_436))


# ---------- GRCh38 (hg38) ----------
_GRCH38_REGIONS: dict[str, list[tuple[int, int]]] = {
    "chr1":  [(122_026_460, 124_932_724)],
    "chr2":  [(92_188_146, 94_090_557)],
    "chr3":  [(90_772_459, 93_655_574)],
    "chr4":  [(49_708_101, 51_743_951)],
    "chr5":  [(46_485_901, 50_059_807)],
    "chr6":  [(58_553_889, 59_829_934),
              # MHC / HLA region
              (28_510_120, 33_480_577)],
    "chr7":  [(58_169_654, 60_828_234)],
    "chr8":  [(44_033_745, 45_338_887)],
    "chr9":  [(43_236_168, 45_518_558)],
    "chr10": [(39_686_683, 41_593_521)],
    "chr11": [(51_078_349, 54_425_074)],
    "chr12": [(34_769_408, 37_185_252)],
    "chr13": [(16_000_001, 18_051_248)],
    "chr14": [(16_000_001, 18_173_523),
              # IGH – immunoglobulin heavy chain
              (105_586_437, 106_879_844)],
    "chr15": [(17_083_674, 19_725_254)],
    "chr16": [(36_311_159, 38_280_682)],
    "chr17": [(22_813_680, 26_885_980)],
    "chr18": [(15_460_900, 20_861_206)],
    "chr19": [(24_498_981, 27_190_874)],
    "chr20": [(26_436_233, 30_038_348)],
    "chr21": [(10_864_561, 12_915_808)],
    "chr22": [(12_954_789, 15_054_318),
              # IGL – immunoglobulin lambda
              (22_026_076, 22_922_913)],
}
# IGK – immunoglobulin kappa
_GRCH38_REGIONS["chr2"].append((88_857_361, 89_331_083))


# ---------- T2T-CHM13 (CHM13v2.0 / hs1) ----------
# Centromeric satellite arrays are fully resolved and substantially larger
# than the gap placeholders in GRCh37/38.  Coordinates derived from the
# CHM13v2.0 cenSat annotation track.
_T2T_CHM13_REGIONS: dict[str, list[tuple[int, int]]] = {
    "chr1":  [(121_700_000, 143_300_000)],
    "chr2":  [(91_500_000, 94_700_000)],
    "chr3":  [(90_000_000, 96_500_000)],
    "chr4":  [(49_000_000, 54_000_000)],
    "chr5":  [(46_000_000, 51_800_000)],
    "chr6":  [(57_000_000, 60_800_000),
              # MHC / HLA region
              (28_510_120, 33_480_577)],
    "chr7":  [(58_000_000, 62_500_000)],
    "chr8":  [(43_000_000, 47_500_000)],
    "chr9":  [(42_000_000, 67_500_000)],
    "chr10": [(38_500_000, 42_500_000)],
    "chr11": [(50_000_000, 55_500_000)],
    "chr12": [(34_000_000, 37_800_000)],
    "chr13": [(13_000_000, 18_500_000)],
    "chr14": [(10_000_000, 13_800_000),
              # IGH – immunoglobulin heavy chain
              (105_586_437, 106_879_844)],
    "chr15": [(14_500_000, 20_500_000)],
    "chr16": [(35_000_000, 39_000_000)],
    "chr17": [(22_000_000, 27_500_000)],
    "chr18": [(15_000_000, 21_800_000)],
    "chr19": [(24_000_000, 28_500_000)],
    "chr20": [(26_000_000, 31_000_000)],
    "chr21": [(10_000_000, 13_500_000)],
    "chr22": [(12_500_000, 17_500_000),
              # IGL – immunoglobulin lambda
              (22_026_076, 22_922_913)],
}
# IGK – immunoglobulin kappa
_T2T_CHM13_REGIONS["chr2"].append((88_857_361, 89_331_083))


# ── Pseudoautosomal / X-transposed region catalogues ──────────────────────
# PAR1 and PAR2 recombine and segregate like autosomes; XTR is the
# X-transposed region that also shows autosomal-like behaviour.
# Coordinates are 1-based, inclusive (VCF POS convention) and sourced
# from illumina_idat_processing/scripts/utils.sh.

# ---------- GRCh37 (hg19) PAR ----------
_GRCH37_PAR: dict[str, list[tuple[int, int]]] = {
    "X":    [(60_001, 2_699_520),              # PAR1
             (2_699_520, 6_100_000),            # XTR
             (154_931_044, 155_260_560)],        # PAR2
    "Y":    [(10_001, 2_649_520),               # PAR1
             (59_034_050, 59_363_566)],          # PAR2
}

# ---------- GRCh38 (hg38) PAR ----------
_GRCH38_PAR: dict[str, list[tuple[int, int]]] = {
    "chrX": [(10_001, 2_781_479),              # PAR1
             (2_781_479, 6_400_000),            # XTR
             (155_701_383, 156_030_895)],        # PAR2
    "chrY": [(10_001, 2_781_479),              # PAR1
             (56_887_903, 57_217_415)],          # PAR2
}

# ---------- T2T-CHM13 (CHM13v2.0) PAR ----------
_T2T_CHM13_PAR: dict[str, list[tuple[int, int]]] = {
    "chrX": [(0, 2_781_479),                   # PAR1
             (2_781_479, 6_400_875),            # XTR
             (155_701_382, 156_040_895)],        # PAR2
    "chrY": [(0, 2_458_320),                   # PAR1
             (2_458_320, 6_400_875),            # XTR
             (62_122_809, 62_460_029)],          # PAR2
}

# Bare-number variants for GRCh38/T2T PAR (GRCh37 already uses bare names)
_GRCH38_PAR_NOPREFIX = {
    k.replace("chr", ""): v for k, v in _GRCH38_PAR.items()
}
_T2T_CHM13_PAR_NOPREFIX = {
    k.replace("chr", ""): v for k, v in _T2T_CHM13_PAR.items()
}
# GRCh37 uses bare-number by default; create chr-prefixed variant
_GRCH37_PAR_PREFIX = {
    f"chr{k}": v for k, v in _GRCH37_PAR.items()
}

# Master PAR lookups
_BUILD_PAR: dict[str, dict[str, list[tuple[int, int]]]] = {
    "GRCh37": _GRCH37_PAR_PREFIX,
    "GRCh38": _GRCH38_PAR,
    "T2T-CHM13": _T2T_CHM13_PAR,
}

_BUILD_PAR_NOPREFIX: dict[str, dict[str, list[tuple[int, int]]]] = {
    "GRCh37": _GRCH37_PAR,
    "GRCh38": _GRCH38_PAR_NOPREFIX,
    "T2T-CHM13": _T2T_CHM13_PAR_NOPREFIX,
}


# Bare-number (Ensembl-style) variants of each catalogue
_GRCH37_REGIONS_NOPREFIX = {
    k.replace("chr", ""): v for k, v in _GRCH37_REGIONS.items()
}
_GRCH38_REGIONS_NOPREFIX = {
    k.replace("chr", ""): v for k, v in _GRCH38_REGIONS.items()
}
_T2T_CHM13_REGIONS_NOPREFIX = {
    k.replace("chr", ""): v for k, v in _T2T_CHM13_REGIONS.items()
}

# Master lookup (canonical build name → chr-prefixed regions)
_BUILD_REGIONS: dict[str, dict[str, list[tuple[int, int]]]] = {
    "GRCh37": _GRCH37_REGIONS,
    "GRCh38": _GRCH38_REGIONS,
    "T2T-CHM13": _T2T_CHM13_REGIONS,
}

_BUILD_REGIONS_NOPREFIX: dict[str, dict[str, list[tuple[int, int]]]] = {
    "GRCh37": _GRCH37_REGIONS_NOPREFIX,
    "GRCh38": _GRCH38_REGIONS_NOPREFIX,
    "T2T-CHM13": _T2T_CHM13_REGIONS_NOPREFIX,
}

# Supported build names (for user-facing messages)
SUPPORTED_BUILDS = ("GRCh37", "GRCh38", "T2T-CHM13")


def get_exclusion_regions(
    build: str,
    chromosomes: list[str] | None = None,
) -> dict[str, list[tuple[int, int]]]:
    """Return default exclusion regions for a genome build.

    Parameters
    ----------
    build : str
        ``"GRCh37"``, ``"GRCh38"``, or ``"T2T-CHM13"`` (case-insensitive;
        ``"hg19"``, ``"hg38"``, ``"hs1"``, and ``"CHM13"`` are accepted
        as aliases).
    chromosomes : list of str or None
        If provided, the returned regions use matching chromosome naming
        (``chr``-prefixed or bare numbers).

    Returns
    -------
    regions : dict mapping chromosome → list of (start, end)
    """
    canon = _normalise_build(build)
    if canon not in _BUILD_REGIONS:
        raise ValueError(
            f"Unknown genome build {build!r}. "
            f"Supported: {', '.join(SUPPORTED_BUILDS)}."
        )

    # Determine whether to use chr-prefixed or bare names
    use_prefix = True
    if chromosomes:
        use_prefix = any(c.startswith("chr") for c in chromosomes)

    if use_prefix:
        regions = _BUILD_REGIONS[canon]
    else:
        regions = _BUILD_REGIONS_NOPREFIX[canon]

    return dict(regions)


def get_par_regions(
    build: str,
    chromosomes: list[str] | None = None,
) -> dict[str, list[tuple[int, int]]]:
    """Return PAR/XTR regions for a genome build.

    These regions should be excluded from X-chromosome GRM (X-GRM)
    computation because they recombine and segregate like autosomes.

    Parameters
    ----------
    build : str
        ``"GRCh37"``, ``"GRCh38"``, or ``"T2T-CHM13"`` (case-insensitive;
        common aliases accepted).
    chromosomes : list of str or None
        If provided, the returned regions use matching chromosome naming
        (``chr``-prefixed or bare numbers).

    Returns
    -------
    regions : dict mapping chromosome → list of (start, end)
        1-based inclusive coordinates.
    """
    canon = _normalise_build(build)
    if canon not in _BUILD_PAR:
        raise ValueError(
            f"Unknown genome build {build!r}. "
            f"Supported: {', '.join(SUPPORTED_BUILDS)}."
        )

    use_prefix = True
    if chromosomes:
        use_prefix = any(c.startswith("chr") for c in chromosomes)

    if use_prefix:
        regions = _BUILD_PAR[canon]
    else:
        regions = _BUILD_PAR_NOPREFIX[canon]

    return dict(regions)


def _normalise_build(build: str) -> str:
    """Map common aliases to canonical build names."""
    mapping = {
        "grch37": "GRCh37",
        "hg19": "GRCh37",
        "grch38": "GRCh38",
        "hg38": "GRCh38",
        "t2t-chm13": "T2T-CHM13",
        "chm13": "T2T-CHM13",
        "chm13v2.0": "T2T-CHM13",
        "chm13v2": "T2T-CHM13",
        "hs1": "T2T-CHM13",
    }
    return mapping.get(build.lower(), build)


def detect_build(path: str | Path) -> str | None:
    """Attempt to detect the genome build from a BCF/VCF file.

    Detection strategy (in order):

    1. Look for an ``assembly`` tag in contig header lines.
    2. Compare chromosome 1 length against known reference lengths.

    Parameters
    ----------
    path : str or Path
        Path to the BCF or VCF file.

    Returns
    -------
    build : str or None
        ``"GRCh37"``, ``"GRCh38"``, or ``"T2T-CHM13"`` if detected,
        otherwise ``None``.
    """
    vcf = pysam.VariantFile(str(path))
    header = vcf.header

    # Strategy 1: check contig header text for known build identifiers
    _ALIASES = (
        "T2T-CHM13", "CHM13", "hs1",
        "GRCh38", "hg38",
        "GRCh37", "hg19",
    )
    for contig in header.contigs:
        rec = header.contigs[contig]
        header_str = str(rec.header_record)
        for alias in _ALIASES:
            if alias in header_str:
                vcf.close()
                build = _normalise_build(alias)
                logger.info("Detected build %s from contig header", build)
                return build

    # Strategy 2: match chromosome 1 length
    for name in ("chr1", "1"):
        if name in header.contigs:
            length = header.contigs[name].length
            if length is not None and length > 0:
                for build_name, expected in _CHR1_LENGTHS.items():
                    if abs(length - expected) < _LENGTH_TOLERANCE:
                        vcf.close()
                        logger.info(
                            "Detected build %s from chr1 length (%d)",
                            build_name,
                            length,
                        )
                        return build_name

    vcf.close()
    logger.warning("Could not detect genome build from %s", path)
    return None
