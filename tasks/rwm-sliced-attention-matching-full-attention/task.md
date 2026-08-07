## Context

The scaled dot‑product attention mechanism computes  

$$\text{Attention}(Q, K, V)=\text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V,$$

where $Q\in\mathbb{R}^{n_q\times d}$, $K\in\mathbb{R}^{n_k\times d}$, $V\in\mathbb{R}^{n_k\times d_v}$, and the softmax is applied row‑wise.

The intermediate score matrix $S=QK^\top$ requires memory proportional to $n_q n_k$, which becomes prohibitive for long sequences.  
Because the softmax is row‑independent (the output for a query depends only on its own row of $S$), we can split $Q$ into chunks, compute attention for each chunk separately, and concatenate the results.  The memory footprint per chunk is limited to $\text{chunk size}\times n_k$, reducing overall peak memory.

## Task

Implement the function  

```python
def chunked_attention(Q: list[list[float]], K: list[list[float]], V: list[list[float]], chunk_size: int) -> tuple[list[list[float]], int]:
```

It takes arrays `Q` (shape $(n_q,d)$), `K` (shape $(n_k,d)$), `V` (shape $(n_k,d_v)$), and an integer `chunk_size`.  
It must return a tuple `(output, peak_bytes)` where:

- `output` is the $(n_q,d_v)$ result of scaled dot‑product attention, **numerically identical** to the full (non‑chunked) computation.
- `peak_bytes` is the maximum memory (in bytes) occupied by any single score matrix chunk, i.e. `chunk_size * n_k * 8` (each element is an `float64`).

You must implement the attention by iterating over chunks of the query dimension.  Do **not** compute the full $n_q\times n_k$ score matrix.

## Example

```python
Q = [[1.0, 0.0], [0.0, 0.0]]
K = [[1.0, 0.0], [1.0, 0.0]]
V = [[1.0], [2.0]]
out, peak = chunked_attention(Q, K, V, chunk_size=1)
print(out)   # [[1.5], [1.5]]
print(peak)  # 16   (1 * 2 * 8)
```

## What the gate checks

Two gates:

1. **max_abs_err** ≤ $10^{-6}$ – the output must match the full attention result within a tight absolute tolerance.
2. **peak_bytes_acc** == 1.0 – the reported `peak_bytes` must exactly equal `chunk_size * K.shape[0] * 8` for every test case.
