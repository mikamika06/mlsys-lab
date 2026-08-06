# Deduplicate and Length-Balance a Calibration Set

Quantization calibration pipelines for Large Language Models (LLMs) often run into severe accuracy degradation when supplied with raw text corpora. In our staging pipeline, post-training quantization (PTQ) runs on raw web dumps produce erratic scale factors. Inspection shows two primary issues:
1. High duplicate density (exact and near-duplicate text spans) causes activation histograms to over-represent specific token sequences, clipping outlier activations across attention and projection layers.
2. Unbalanced sequence length distributions lead to wasteful padding or disproportionate attention mask biases when batching sequence chunks for calibration forward passes.

We need a dedicated preprocessing pipeline that ingests a raw calibration dataset, deduplicates samples based on exact token min-hashes, and balances sequence lengths into structured target buckets prior to PTQ execution.

## Requirements

1. **Deduplication (`calib/dedup.py`)**:
   - Implement `deduplicate_samples(samples, num_perm=128, threshold=0.8)`.
   - Extract character n-grams ($n=5$) or token IDs to generate $K$ MinHash signatures for each sequence.
   - Group candidates using Locality-Sensitive Hashing (LSH) bands and filter out sample pairs whose Jaccard similarity meets or exceeds `threshold`.
   - Retain the earliest sample when duplicates are detected.

2. **Length Balancing (`calib/balance.py`)**:
   - Implement `balance_lengths(samples, bucket_sizes, target_counts)`.
   - Given tokenized input samples, group sequences into length buckets (`bucket_sizes` defining upper bounds).
   - Downsample over-represented buckets and pad or truncate sequences as specified by target distribution counts to produce a length-balanced calibration batch.

3. **Regression Safety (`tests/test_regression.py`)**:
   - Provide unit tests verifying that exact and high-similarity near-duplicate sequences are purged while maintaining unique contents and targeted bucket distributions.
