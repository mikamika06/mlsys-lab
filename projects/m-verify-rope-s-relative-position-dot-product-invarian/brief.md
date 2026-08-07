# RoPE Invariance, Position Interpolation, and Context Extrapolation

A research team scaling a Llama-style model reported unexpected behavior during context window expansion experiments. While evaluating Rotary Position Embedding (RoPE) mechanisms, their custom attention kernel yielded varying inner products depending on absolute positions despite constant relative offsets. Furthermore, initial attempts to extend the model's context window from 2048 to 8192 tokens resulted in severe perplexity degradation when evaluated on long sequences.

You are tasked with diagnosing and implementing the relative-position invariants of RoPE, implementing linear Position Interpolation (PI) for extended context evaluation, and measuring the perplexity shift across context boundaries.

## Target Functionality

1. **Relative Position Invariance**: Verify that applying 2D / multi-D RoPE transformation to query vector $q$ at position $m$ and key vector $k$ at position $n$ yields a dot product dependent solely on relative offset $(m - n)$.
2. **Position Interpolation**: Implement sequence-level position rescaling for RoPE to allow context window extension with minimal perplexity degradation.
3. **Perplexity Degradation Analysis**: Compute cross-entropy loss and perplexity across sequence lengths to quantify out-of-distribution context blow-up compared to interpolated position embeddings.

Inspect the codebase structure, implement the missing mathematical functions in `rope/core.py`, and supply regression tests in `tests/test_regression.py`.
