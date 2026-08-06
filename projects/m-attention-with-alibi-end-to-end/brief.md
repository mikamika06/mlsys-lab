# Ticket: ALiBi Attention Discrepancies and Numerical Instabilities

## Symptom
Downstream evaluation of our long-context models shows erratic sequence outputs when applying Attention with Linear Biases (ALiBi) across different attention backends. Numerical logs indicate unexpected score explosions and overflow issues during long-sequence inference when softcapping is bypassed or misconfigured across supporting backends.

## Tasks
1. Implement end-to-end multi-head attention supporting ALiBi bias offsets, custom scaling, and optional logit softcapping in `alibi_attn/attention.py`.
2. Construct a modifier/backend support matrix helper in `alibi_attn/matrix.py` that validates whether a requested combination of attention modifiers (e.g., ALiBi, Softcap, Causal Masking, Sliding Window) is supported by target backend implementations (Standard PyTorch, FlashAttention, PagedAttention).
3. Implement an overflow rate calculation tool in `alibi_attn/overflow.py` that measures logit value overflow ratios under softcapped vs unsoftcapped conditions.
4. Write a safeguard test suite in `tests/test_regression.py` verifying that attention outputs match reference ALiBi calculations and detecting failures when softcapping is disabled on explosive logit scores.
