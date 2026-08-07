## Context

Attention implementations often add several score modifiers before applying softmax. An additive bias such as ALiBi can be written as a matrix $B$ that is added to the raw attention scores $S$:

$$
S' = S + B .
$$

The numerically stable softmax subtracts the maximum value from each row before exponentiation:

$$
\mathrm{softmax}(x_i) =
\frac{\exp(x_i - m)}{\sum_j \exp(x_j - m)},
\qquad
m = \max_j x_j .
$$

For any row-wise constant $c$, subtracting $c$ does not change the softmax because:

$$
\frac{\exp(x_i-c)}{\sum_j \exp(x_j-c)}
=
\frac{\exp(x_i)\exp(-c)}{\sum_j \exp(x_j)\exp(-c)}
=
\frac{\exp(x_i)}{\sum_j \exp(x_j)} .
$$

This allows additive score modifiers to be incorporated while computing the same max-subtracted softmax. A production attention kernel can combine modifiers such as ALiBi, local window masking, and soft-cap transformations without materializing unnecessary intermediate tensors.

## Task

Implement `fused_attention_scores(scores, alibi, window, soft_cap)`:

```python
def fused_attention_scores(scores, alibi, window, soft_cap):
    ...
```

The inputs are:

- `scores`: a 2-D floating point array of shape $(n, n)$ containing raw attention scores.
- `alibi`: a 2-D floating point array of shape $(n, n)$ containing additive ALiBi score modifiers.
- `window`: a non-negative integer. Positions with $|i-j| > \text{window}$ are masked.
- `soft_cap`: a positive scalar. Before softmax, scores are soft-capped using:

$$
x \leftarrow \text{soft\_cap} \cdot \tanh(x / \text{soft\_cap}) .
$$

Return the final attention probability matrix as `float64`.

The mathematical reference order is:

1. Add ALiBi: $x = scores + alibi$.
2. Apply the soft-cap transformation.
3. Apply the window mask by setting masked positions to $-\infty$.
4. Apply row-wise max-subtracted softmax in `float64`.

Your implementation may combine these operations into a fused computation, but the numerical result must match the reference.

## Example

```python

scores = [[1.0, 2.0], [0.5, 0.0]]
alibi = [[0.0, -0.2], [0.1, 0.0]]

out = fused_attention_scores(scores, alibi, window=1, soft_cap=2.0)

# out contains the two row-wise attention distributions.
```

## What the gate checks

The gate computes a Python oracle by applying the documented sequence of score modifiers and stable softmax operations in `float64`.

The returned matrix is compared with the oracle using the maximum absolute error:

$$
\max_{i,j}|A_{ij} - A^{ref}_{ij}|.
$$

The value must be less than or equal to $10^{-5}$. Implementations that omit a modifier, apply the window incorrectly, or perform an unstable softmax will fail.
