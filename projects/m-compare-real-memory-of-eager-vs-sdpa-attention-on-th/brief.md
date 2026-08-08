# Symptom: Fine-Tuning Long Contexts OOMs on Small Models During Backward Pass

Our ML platform team received reports that training runs on a standard 12-layer Transformer model (e.g., GPT-2 scale) consistently crash with GPU Out-Of-Memory (OOM) errors as sequence length scales past 2048, even when batch size is kept at 1. Oddly, switching the attention layer from standard Eager PyTorch Attention to Scaled Dot-Product Attention (SDPA / FlashAttention kernel behavior) allows sequence length 8192 to complete smoothly on the exact same GPU hardware without OOM.

Engineers suspect the discrepancy comes from activation memory retained for the backward pass and intermediate activation spikes, but our current memory planning tools only estimate static model weight sizes. We need an accurate activation accounting tool that parses layer trace metadata and exact execution shapes to quantify real activation memory differences between Eager and SDPA attention on the target model.

## Your Task

1. Implement `memacc/accounting.py` to parse execution trace configurations and calculate peak memory and retained backward activation footprints for Eager MHA vs. fused SDPA attention.
2. Implement `memacc/compare.py` to analyze whole-model memory across varying sequence lengths and compute the empirical size ratio between Eager and SDPA activation memory footprints.
3. Write `tests/test_regression.py` as a safeguard that verifies memory scaling laws and fails if SDPA activation accounting erroneously includes quadratic attention matrices ($O(S^2)$).
