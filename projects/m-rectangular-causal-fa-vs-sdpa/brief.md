# Rectangular Causal Attention: FlashAttention vs SDPA Alignment Bug

## Symptom
Our high-throughput inference engine observed unexpected output drift and subtle generation quality degradation when serving models with prefix caching enabled. Specifically, during prompt processing where the key/value sequence length $N_{kv}$ exceeds the query sequence length $N_q$ (rectangular causal attention, e.g., $N_q = 64, N_{kv} = 256$), switching the backend from PyTorch's `scaled_dot_product_attention` (SDPA) to a custom FlashAttention (FA) implementation causes numerical divergence ($> 10^{-2}$ absolute difference in output logits).

Investigations show that when sequence lengths differ, standard causal masks can be aligned differently relative to the query/key matrix boundaries. SDPA defaults to top-left mask alignment (where query $i$ attends to keys $j \le i$), whereas FlashAttention-style kernels natively default to bottom-right alignment (where query index $i$ corresponds to key index $j \le i + (N_{kv} - N_q)$). When caching prompt prefixes, this offset mismatch causes queries to attend to incorrect, unmasked keys or falsely mask out valid historical tokens.

## Task
You must diagnose and fix this alignment bug across three distinct milestones:

1. **Mask-Alignment Probe (`rectatt/probe.py`)**: Implement an alignment probe and mask generation logic that computes explicit boolean causal masks for both top-left and bottom-right alignment modes. Identify the exact query-key index mapping differences under rectangular shapes ($N_q \ne N_{kv}$).
2. **FA vs SDPA Causal Attention (`rectatt/attention.py`)**: Implement reference attention routines for both PyTorch SDPA and FlashAttention kernel simulation. Ensure that rectangular causal queries produce identical outputs across both backends by adjusting the offset alignment parameter when converting query/key sequence bounds.
3. **Regression Safeguard (`tests/test_regression.py`)**: Construct a comprehensive regression test suite that verifies mask offsets, checks numerical output parity across backend switches for asymmetric sequence shapes, and catches improper top-left vs bottom-right mask alignment regressions.
