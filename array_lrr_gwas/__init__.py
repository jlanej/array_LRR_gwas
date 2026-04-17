"""array_lrr_gwas: Batch effect correction and LMM GWAS for array-based LRR values."""

from array_lrr_gwas.subsetting import subset_markers, select_every_nth
from array_lrr_gwas.decomposition import rsvd, decompose, estimate_rsvd_marker_budget
from array_lrr_gwas.correction import correct_lrr, qr_precompute, residualize_qr
from array_lrr_gwas.select_k import select_k_mp, select_k_elbow
from array_lrr_gwas.genome_build import detect_build, get_exclusion_regions
from array_lrr_gwas.association import run_association, AssociationResult
from array_lrr_gwas.segmentation import segment_associations, SegmentationResult
from array_lrr_gwas.genotypes import read_genotypes
from array_lrr_gwas.grm import compute_grm
from array_lrr_gwas.sample_sheet import read_sample_sheet, align_samples
from array_lrr_gwas.qc_config import load_config as load_qc_config
from array_lrr_gwas.variant_qc import read_collated_variant_qc, variant_qc_mask
from array_lrr_gwas.gwas_report import (
    generate_gwas_report,
    lambda_gc,
    top_hits,
    annotate_hits_with_genes,
    load_gene_table,
)

__all__ = [
    "subset_markers",
    "select_every_nth",
    "rsvd",
    "decompose",
    "estimate_rsvd_marker_budget",
    "correct_lrr",
    "qr_precompute",
    "residualize_qr",
    "select_k_mp",
    "select_k_elbow",
    "detect_build",
    "get_exclusion_regions",
    "run_association",
    "AssociationResult",
    "segment_associations",
    "SegmentationResult",
    "read_genotypes",
    "compute_grm",
    "read_sample_sheet",
    "align_samples",
    "load_qc_config",
    "read_collated_variant_qc",
    "variant_qc_mask",
    "generate_gwas_report",
    "lambda_gc",
    "top_hits",
    "annotate_hits_with_genes",
    "load_gene_table",
]
