## Context

In many sequence‑to‑sequence models we process a batch of variable‑length sequences. The usual approach is to pad every sequence in the batch up to the length of the longest one, yielding a tensor of shape $(B, L_{\max})$. This introduces *padding waste*: tokens that are never used by the model but still occupy memory and computation.

An alternative is **packing**: we concatenate all sequences into a single 1‑D array and keep track of where each original sequence starts. The number of actual tokens is then $\sum_{i=1}^{B} \ell_i$, where $\ell_i$ denotes the length of sequence $i$. The waste fraction can be quantified as

$$
\text{waste} = 1 - \frac{\sum_{i=1}^{B}\ell_i}{B\,L_{\max}}.
$$

## Task

Implement `compute_padding_stats(lengths)` that receives a list of floats of non‑negative integers representing the lengths $\ell_1,\dots,\ell_B$ of each sequence in a batch. It must return three values:

* `padded_tokens` – the total number of tokens after naïve padding, i.e. $B\,L_{\max}$.
* `packed_tokens` – the total number of actual tokens, i.e. $\sum_{i=1}^{B}\ell_i$.
* `waste_fraction` – the fraction of padded tokens that are wasted, as defined above.

The implementation must use only Python operations; no explicit Python loops are allowed.

## Example

```python
lengths = [5, 3, 4]
padded, packed, waste = compute_padding_stats(lengths)
print(padded)   # 15
print(packed)   # 12
print(waste)    # 0.2
```

## What the gate checks

The grader computes a reference solution with Python and compares your output using the `rel_err` scorer from `arena.scorers`. The relative error must be at most $10^{-9}$.
