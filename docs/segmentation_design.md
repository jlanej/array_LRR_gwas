# Segmentation Design

Post-GWAS segmentation defines CNV-associated genomic intervals from
per-marker association results.  This module consumes the TSV output of
the `associate` sub-command and produces BED-format intervals with
provenance statistics.

See also: `docs/association_engine_design.md` § *Output Contract for
Downstream Segmentation*.

## Strategies

Two segmentation strategies are provided.  Both operate per-chromosome
to prevent state carry-over across chromosome boundaries.

### 1. Two-State HMM (`--strategy hmm`, default)

A Hidden Markov Model partitions markers into *null* (state 0) and
*associated* (state 1) states.

**Emission model.**  Observations are `-log10(p)` for each marker.
Under the null hypothesis, p-values are Uniform(0, 1), so `-log10(p)`
follows an Exponential distribution with rate `ln(10) ≈ 2.303`.  The
associated state uses a smaller rate (default 0.1), expecting larger
`-log10(p)` values (mean ~10).

| Parameter         | CLI flag           | Default   | Meaning |
|-------------------|--------------------|-----------|---------|
| `null_rate`       | `--null-rate`      | `ln(10)`  | Exponential rate for null emission |
| `signal_rate`     | `--signal-rate`    | `0.1`     | Exponential rate for associated emission |
| `prior_assoc`     | `--prior-assoc`    | `0.001`   | Prior probability of associated state |
| `transition_prob` | `--transition-prob` | `1e-4`   | Per-marker state-transition probability |

**Transition matrix.**  Symmetric off-diagonal probability
`transition_prob`.  This encodes the prior belief that state changes
are rare relative to the marker spacing.

**Decoding.**  The Viterbi algorithm finds the most likely state
sequence in O(T) time and O(T) space.  Contiguous runs of state 1 are
collapsed into intervals.

### 2. Threshold-and-Merge (`--strategy threshold`)

A simpler baseline:

1. Flag markers with `p_value < p_threshold` (default 5 × 10⁻⁸).
2. Merge contiguous flagged runs into intervals.
3. Merge intervals separated by ≤ `max_gap` bp (default 1 Mb).

| Parameter     | CLI flag        | Default   |
|---------------|-----------------|-----------|
| `p_threshold` | `--p-threshold` | `5e-8`    |
| `max_gap`     | `--max-gap`     | `1000000` |

## Output Format

BED-format (0-based, half-open coordinates) with additional columns:

| Column         | Type    | Description |
|----------------|---------|-------------|
| `chrom`        | `str`   | Chromosome identifier |
| `start`        | `int`   | 0-based inclusive start position |
| `end`          | `int`   | 0-based exclusive end position |
| `name`         | `str`   | Region identifier (`region_1`, `region_2`, …) |
| `n_markers`    | `int`   | Number of contributing markers |
| `min_p`        | `float` | Minimum p-value in segment |
| `mean_beta`    | `float` | Mean effect-size estimate |
| `max_abs_stat` | `float` | Maximum absolute test statistic |
| `method`       | `str`   | Source association method |

The header line is prefixed with `#`.

## Provenance

`SegmentationResult.parameters` records the strategy-specific
parameters used for a given run, enabling full reproducibility.

## Edge Cases

- **Empty input:** Produces an empty BED (header only).
- **No significant markers:** Produces an empty BED.
- **p = 0:** Clipped to `1e-300` before log-transform.
- **Single marker:** Valid segment with `start = pos - 1`, `end = pos`.
- **Multiple chromosomes:** Segmented independently.

## CLI Usage

```
array-lrr-gwas segment results.tsv -o regions.bed
array-lrr-gwas segment results.tsv -o regions.bed --strategy threshold
array-lrr-gwas segment results.tsv -o regions.bed --strategy hmm \
    --signal-rate 0.05 --prior-assoc 0.01
```
