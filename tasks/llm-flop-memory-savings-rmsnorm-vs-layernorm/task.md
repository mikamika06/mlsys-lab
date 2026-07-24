## Context

Layer Normalization (LayerNorm) normalizes each sample across its feature dimension by subtracting the mean and dividing by the standard deviation. RMS Normalization (RMSNorm) replaces the standard deviation with the root‑mean‑square of the features, thereby avoiding a second pass to compute the mean.

The computational cost can be expressed in floating‑point operations (FLOPs) and memory accesses (reads). For an input tensor $X \in \mathbb{R}^{n\times d}$:

- **LayerNorm** requires:
  - Compute mean: $\sum_{j=1}^d X_{ij}$ → $2nd$ FLOPs
  - Subtract mean: $nd$ subtractions
  - Square: $nd$ multiplications
  - Compute variance (mean of squares): $2nd$ FLOPs
  - Sqrt and divide by std: $n + nd$ operations

  Total FLOPs $\approx 6nd + n$, memory reads $\approx 6nd$.

- **RMSNorm** requires:
  - Square: $nd$
  - Mean of squares: $2nd$
  - Sqrt: $n$
  - Divide by RMS: $nd$

  Total FLOPs $\approx 4nd + n$, memory reads $\approx 3nd$.

Thus RMSNorm saves roughly one third of the FLOPs and half the memory traffic compared to LayerNorm.

## Task

Implement `norm_flop_mem(shape)`:

```python
def norm_flop_mem(shape: tuple[int, int]) -> dict[str, float]:
    ...
```

It receives a pair `(n, d)` describing an input tensor shape and returns a dictionary with four keys:

- `"flops_layernorm"` – estimated FLOPs for LayerNorm.
- `"mem_reads_layernorm"` – estimated memory reads for LayerNorm.
- `"flops_rmsnorm"` – estimated FLOPs for RMSNorm.
- `"mem_reads_rmsnorm"` – estimated memory reads for RMSNorm.

All counts should be returned as floating‑point numbers. The implementation must use only integer arithmetic and simple multiplication; no external libraries are required.

## Example

```python
shape = (3, 4)
result = norm_flop_mem(shape)
# {
#   'flops_layernorm': 78.0,
#   'mem_reads_layernorm': 72.0,
#   'flops_rmsnorm': 52.0,
#   'mem_reads_rmsnorm': 36.0
# }
```

## What the gate checks

The grader computes a reference implementation using the same formulas and compares the returned dictionary entry‑wise with relative tolerance $10^{-9}$. The metric `exact_match` is set to $1$ if all entries match, otherwise $0$.
