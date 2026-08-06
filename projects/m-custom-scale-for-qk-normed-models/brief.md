# QK-Normed Attention Custom Scaling Bug

## Symptom
Our high-throughput inference engine for QK-normed Transformer models (such as Cohere Command-R / Gemma style architectures with LayerNorm or RMSNorm applied directly to Query and Key projections) is producing corrupt output distributions and high perplexity spikes when custom scaling factors are enabled during FlashAttention dispatch.

When standard attention computes $\text{softmax}(QK^T / \sqrt{d})$, QK-normalization changes the numerical dynamic significantly. In QK-normed models, queries and keys are unit-normed along the head dimension before applying attention scale factors. Some serving configurations override the implicit default scaling factor ($\frac{1}{\sqrt{d}}$ or custom scale constants like $\frac{1}{d_k}$) via an explicit `scale` argument to FlashAttention. However, our current scaled attention runtime applies scaling inconsistently across key normalization and scale pre-computations, leading to out-of-range softmax logits and inaccurate sequence scores.

## Task
You need to inspect the attention calculation layer and rectify the custom scale handling for QK-normed models:

1. Implement core matrix scoring with explicit query-key normalization and arbitrary scale factor support.
2. Ensure logit soft-capping and custom scaling factors interact correctly without double-scaling or missing normalization factors.
3. Add a regression test suite in `tests/test_regression.py` that verifies attention output fidelity under modified scaling and catches common mistakes such as dropping custom scale overrides when QK-normalization is active.
