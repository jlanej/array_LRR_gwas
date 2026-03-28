FROM python:3.12-slim

LABEL maintainer="jlanej" \
      description="array_lrr_gwas: Batch-effect correction for array-based LRR values" \
      license="GPL-3.0-or-later"

# Install system dependencies required by pysam / htslib
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        libc6-dev \
        zlib1g-dev \
        libbz2-dev \
        liblzma-dev \
        libcurl4-openssl-dev \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY pyproject.toml README.md LICENSE ./
COPY array_lrr_gwas/ array_lrr_gwas/

# Install the package
RUN pip install --no-cache-dir .

# Verify the installation
RUN python -c "import array_lrr_gwas; print('array_lrr_gwas imported successfully')"

ENTRYPOINT ["array-lrr-gwas"]
CMD ["--help"]
