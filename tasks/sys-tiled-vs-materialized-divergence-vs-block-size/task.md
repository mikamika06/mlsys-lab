## Context

Scaled dot‑product attention is the core of transformer models.  
For a query matrix $Q \in \mathbb{R}^{n\times d}$, key matrix $K$, and value matrix $V$,
the dense forward pass computes

$$
\operatorname{Attn}(Q,K,V) = 
\underbrace{\operatorname{softmax}\!\left(\frac{Q K^\top}{\sqrt{d}}\right)}_{\text{attention weights}}
\, V .
$$

A memory‑efficient implementation, often called *tiled* or *block‑wise*, processes the queries in blocks of size $b$.  
For each block $\mathcal{B}$ it computes a partial contribution

$$
S_{\mathcal{B}} = 
\operatorname{softmax}\!\left(\frac{Q_{\mathcal{B}} K^\top}{\sqrt{d}}\right)\! V ,
$$

and accumulates these into the final output.  
Because floating‑point addition is not associative, the order in which partial results are summed can introduce a small numerical drift compared to the fully materialised dense computation.

## Task

Implement `attention_divergence(block_sizes, seq_len=128, d_model=64)`:

```python
def attention_divergence(block_sizes: list[int], seq_len: int=128, d_model: int=64) -> list[float]:
    ...
```

The function should generate a fixed random query/key/value tensor of shape `(seq_len, d_model)` using `random.seed(0)` and the standard normal distribution.
For each block size in `block_sizes` it must compute the tiled attention output as described above and compare it to the dense implementation.  
It should return a list containing the **maximum absolute difference** between the two outputs for every block size.

The returned array must have type `float64`, shape `(len(block_sizes),)`, and be deterministic across runs.

## Example

```python
from solution_ref import attention_divergence

errs = attention_divergence([16, 32, 64], seq_len=128, d_model=64)
print(errs)
# array([2.3e-06, 1.8e-06, 9.5e-07])
```

The numbers will be small because the two implementations differ only by floating‑point accumulation order.

## What the gate checks

Two metrics are evaluated:

* `max_abs_err` – the maximum absolute difference between your output and the reference implementation’s output. It must not exceed $10^{-5}$.
* (Implicit) Monotonicity – the errors should not increase when the block size grows, but this is only a sanity check and does not affect grading.

The grader uses Python to recompute both dense and tiled attention for the same random tensors; no hard‑coded values are used. A correct implementation will pass all gates.
