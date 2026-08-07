## Context

In transformer inference the *prefill* stage loads a sequence of tokens into a cache before any attention is computed.  
When an *Attention Pattern Cache* (APC) hits, the model can reuse previously computed attention patterns and avoid re‑computing the matrix multiplications that would otherwise be performed for each token.  The computational savings are measured in floating‑point operations (FLOPs).  

If a token is reused \(r_i\) times across all layers, and each reuse saves \(f\) FLOPs per token, then the total FLOPs saved by APC for a batch of sequences is

$$
S = f \sum_{i} r_i .
$$

The goal of this task is to implement a function that computes \(S\).

## Task

Implement `flops_saved_by_apc`:

```python
def flops_saved_by_apc(reused_counts: list[int], per_token_flop: float) -> float:
    ...
```

* `reused_counts` – a 1‑D list of non‑negative integers; each element is the number of times a particular token was reused.
* `per_token_flop` – a scalar floating point value representing the FLOPs saved per reuse of a single token.

The function must return the total FLOPs saved as a Python `float`.  Use only Python operations; no explicit Python loops are required but not forbidden.

## Example

```python
counts = [3, 5, 2]          # three tokens reused 3, 5 and 2 times respectively
f = 1000.0                            # each reuse saves 1000 FLOPs
saved = flops_saved_by_apc(counts, f)
print(saved)                          # 10000.0
```

## What the gate checks

The grader evaluates the relative error of your result against a Python oracle:

$$
\mathrm{rel\_err} = \frac{\lvert \hat S - S_{\text{ref}} \rvert}
                         {\lvert S_{\text{ref}} \rvert + 10^{-12}}
$$

The gate passes if `rel_err <= 1e-9`.  The reference value is computed by the grader itself; no hard‑coded constants are used.
