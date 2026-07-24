## Context

The NF4 scheme quantizes real numbers into 16 discrete levels derived from the standard normal distribution. The codebook $\mathcal{L} = \{\ell_0,\dots,\ell_{15}\}$ is obtained by taking the $(k+0.5)/16$‑quantiles of a large sample from $\mathcal{N}(0,1)$. For a block of values $x$, we first compute its absolute maximum
$$m=\max_i |x_i|,$$
normalize $y_i=x_i/m$ (with $m=1$ if all entries are zero), and then replace each $y_i$ by the nearest codebook entry $\ell_{k(i)}$. The 4‑bit index $k(i)\in\{0,\dots,15\}$ is stored as a byte. Dequantization simply rescales:
$$\hat{x}_i = \ell_{k(i)}\,m.$$

## Task

Implement the function `nf4_quantize_dequant(x: np.ndarray, blocksize: int = 128) -> Tuple[np.ndarray, np.ndarray]` that takes an arbitrary‑shaped NumPy array of dtype float64 and returns a pair `(codes, deq)`:

- `codes`: a uint8 array containing the 4‑bit indices (0–15). The shape must match the input.
- `deq`: the reconstructed float64 array obtained by multiplying each codebook level by the blockwise absolute maximum.

The implementation must be fully vectorised; no explicit Python loops over elements. It should handle blocks of size `blocksize` and correctly process the last incomplete block.

## Example

```python
import numpy as np
x = np.array([0.0, 1.5, -2.3, 0.7])
codes, deq = nf4_quantize_dequant(x)
print(codes)   # e.g., [0, 12, 3, 9]
print(deq)     # reconstructed values close to the original
```

## What the gate checks

Two gates are applied:

1. **exact_match** – The byte‑wise equality of the returned `codes` array with a reference implementation that uses the exact NF4 codebook. A mismatch yields a score of 0.
2. **max_abs_err** – The maximum absolute difference between the student's `deq` and the reference reconstruction must not exceed $10^{-12}$.

Both gates must pass for the solution to be accepted.
