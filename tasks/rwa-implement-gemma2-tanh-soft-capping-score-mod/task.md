## Context

Attention computes scores between query vectors $Q$ and key vectors $K$ before applying a softmax normalization.

For a query matrix $Q \in \mathbb{R}^{n \times d}$ and key matrix $K \in \mathbb{R}^{m \times d}$, the scaled attention scores are

$$S = \frac{QK^\top}{\sqrt{d}}.$$

Gemma 2 applies a soft-capping transformation to these scores before softmax. Given a positive cap value $c$, each score is modified as

$$S'_{ij} = c \tanh\left(\frac{S_{ij}}{c}\right).$$

This keeps extreme logits bounded while preserving the ordering of scores. The final attention output is computed as

$$\mathrm{Attention}(Q,K,V) = \mathrm{softmax}(S')V.$$

The softmax is computed row-wise:

$$\mathrm{softmax}(x_i)=\frac{e^{x_i}}{\sum_j e^{x_j}}.$$

## Task

Implement `attention_with_score_mod(Q, K, V, cap)`:

```python
def attention_with_score_mod(Q: list[list[float]], K: list[list[float]], V: list[list[float]], cap: float) -> list[list[float]]:
    ...
```

The inputs are list:
- `Q` has shape $(n,d)$.
- `K` has shape $(m,d)$.
- `V` has shape $(m,h)$.
- `cap` is a positive floating point scalar.

Return the attention output as a `float64` list of shape $(n,h)$.

The implementation should apply the tanh soft-capping operation to the scaled attention scores before softmax. Use Python operations rather than Python loops.

## Example

```python

Q = [[1.0, 0.0]]
K = [[1.0, 0.0], [0.0, 1.0]]
V = [[2.0], [4.0]]

out = attention_with_score_mod(Q, K, V, 1.0)
```

The output is the weighted value aggregation after soft-capping the two attention logits.

## What the gate checks

The gate computes a Python reference implementation of Gemma2 tanh score soft-capping in `float64`:

$$S' = c\tanh(S/c), \quad S = QK^\top/\sqrt{d}.$$

It compares the submitted function output against the oracle output using the maximum absolute error metric. The error must be less than $10^{-5}$ for multiple inputs and cap values.
