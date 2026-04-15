"""Test helpers for array_lrr_gwas."""


def mock_associate_io(monkeypatch, lrr, samples, variants):
    """Patch BCF I/O functions in ``_run_associate`` with in-memory data.

    Replaces ``read_variant_metadata``, ``stream_lrr_chunks``, and
    ``stream_lrr_contig_chunks`` so that tests can run without a real BCF file.
    """
    import numpy as np  # noqa: F401 (keep unused import for potential use in helpers)

    monkeypatch.setattr(
        "array_lrr_gwas.io_vcf.read_variant_metadata",
        lambda _p: (list(samples), list(variants)),
    )

    def _fake_stream(path, *, chunk_size=5000, sample_mask=None, variant_mask=None):
        _lrr = lrr.copy()
        if sample_mask is not None:
            _lrr = _lrr[:, sample_mask]
        if variant_mask is not None:
            _vars = [v for v, m in zip(variants, variant_mask) if m]
            _lrr = _lrr[variant_mask]
        else:
            _vars = list(variants)
        for start in range(0, _lrr.shape[0], chunk_size):
            end = min(start + chunk_size, _lrr.shape[0])
            yield _lrr[start:end], _vars[start:end]

    monkeypatch.setattr(
        "array_lrr_gwas.io_vcf.stream_lrr_chunks",
        _fake_stream,
    )

    def _fake_contig_stream(path, contigs, *, chunk_size=5000, sample_mask=None):
        """Return only variants whose chrom matches one of the requested contigs."""
        _contig_set = set(contigs)
        _lrr = lrr.copy()
        if sample_mask is not None:
            _lrr = _lrr[:, sample_mask]
        _sel_idx = [
            i for i, v in enumerate(variants)
            if v.get("chrom", "") in _contig_set
        ]
        if not _sel_idx:
            return
        _sel_lrr = _lrr[_sel_idx]
        _sel_vars = [variants[i] for i in _sel_idx]
        for start in range(0, _sel_lrr.shape[0], chunk_size):
            end = min(start + chunk_size, _sel_lrr.shape[0])
            yield _sel_lrr[start:end], _sel_vars[start:end]

    monkeypatch.setattr(
        "array_lrr_gwas.io_vcf.stream_lrr_contig_chunks",
        _fake_contig_stream,
    )
