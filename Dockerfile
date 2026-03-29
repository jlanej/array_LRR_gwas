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
        wget \
    && rm -rf /var/lib/apt/lists/*

# Install plink2 (used by --ld-backend plink2 for fast LD pruning)
RUN wget -qO /usr/local/bin/plink2 \
    "https://s3.amazonaws.com/plink2-assets/alpha6/plink2_linux_x86_64_20250109.zip" \
    && apt-get update && apt-get install -y --no-install-recommends unzip \
    && cd /tmp && unzip /usr/local/bin/plink2 \
    && mv /tmp/plink2 /usr/local/bin/plink2 \
    && chmod +x /usr/local/bin/plink2 \
    && apt-get purge -y unzip wget && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /tmp/*

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
