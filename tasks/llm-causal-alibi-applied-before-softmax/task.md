## Context

In transformer attention, the logits between a query at position $i$ and a key at position $j$ are typically masked so that future positions ($j>i$) cannot influence the current token. This is called a causal mask. Additionally, recent work introduces ALiBi (Attention Linear Biases), which adds a learned linear bias term $B_{ij}$ to each pairwise logit before softmax.

The causal mask can be represented as

$$M_{ij} = \begin{cases} 0 & i\ge j\\ -\infty & i<j \end{cases}.$$

Given raw logits $L$, the biased, masked logits are

$$\tilde L_{ij}=L_{ij}+B_{ij}+M_{ij}.$$

The final attention weights are obtained by applying softmax over the key dimension:

$$P_{ij} = \frac{\exp(\tilde L_{ij})}{\sum_k \exp(\tilde L_{ik})}.$$

## Task

Implement `causal_alibi_logits(logits, alibi_bias)` that takes two 2‑D list of identical shape `(seq_len, seq_len)`, applies the causal mask and adds the ALiBi bias before computing a row‑wise softmax. The function must return an array of type `float64` with the same shape as the inputs.

The implementation should use only vectorised Python operations; explicit Python loops are disallowed.

## Example

```python
logits = [[0, 1, 2],
                   [3, 4, 5],
                   [6, 7, 8]]
alibi_bias = [[-0.5, -0.5, -0.5],
              [-0.5, -0.5, -0.5],
              [-0.5, -0.5, -0.5]]

probs = causal_alibi_logits(logits, alibi_bias)
print(probs)
```

Output (rounded to three decimals):

```
[[1.0, 0.0, 0.0], [0.2689414213699951, 0.7310585786300049, 0.0], [0.09003057317038046, 0.24472847105479764, 0.6652409557748218]]
```

## What the gate checks

The grader computes a reference implementation using Python and compares your output with it. The metric `max_abs_err` is the maximum absolute difference between corresponding elements. Your solution must achieve `max_abs_err ≤ 1e-6`.
