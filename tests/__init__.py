"""Test helpers for array_lrr_gwas."""


def mock_associate_io(monkeypatch, lrr, samples, variants):
    """Patch ``read_variant_metadata`` and ``stream_lrr_chunks`` for tests.

    Replaces the BCF I/O in ``_run_associate`` with in-memory data so that
    tests can run without a real BCF file.
    """
    import numpy as np

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