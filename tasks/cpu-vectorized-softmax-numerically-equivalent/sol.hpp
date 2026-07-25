#pragma once

// ============================================================================
// Reconstruct a NUMERICALLY STABLE softmax over n logits:
//
//   m       = max(logits[0..n))
//   probs[i] = exp(logits[i] - m) / sum_j exp(logits[j] - m)
//
// Subtracting the max before exponentiating is mandatory, not optional --
// a batch of real logits can easily contain a value large enough that
// exp(logits[i]) alone overflows to +inf (and inf / inf is NaN), while
// exp(logits[i] - m) never exceeds exp(0) == 1. This is exactly the shape
// a vectorized (SIMD) softmax kernel takes: a lane-wise max reduction,
// then a lane-wise exp + sum, then a lane-wise divide -- every stage
// applies uniformly across all n logits, no per-element branching, which
// is only safe once the max-subtraction keeps every exponent <= 0.
// ============================================================================
void softmax(const float* logits, int n, float* probs);
