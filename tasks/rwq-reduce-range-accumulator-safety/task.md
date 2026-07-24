## Context

In many quantized neural network backends, activations are stored as unsigned 8‑bit integers with a maximum representable value $q_{\max}$. When accumulating these values across a batch into an integer accumulator, the intermediate sums can exceed the limits of a signed 32‑bit integer ($2^{31}-1$). A common mitigation is to reduce the dynamic range by halving $q_{\max}$ from $127$ to $63$, thereby guaranteeing that the peak partial sum stays within bounds for typical batch sizes.

## Task

Implement `reduce_range_accumulator_safety(X)`:

```python
def reduce_range_accumulator_safety(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...
```

- `X` is a 2‑D NumPy array of type `uint8`, shape `(N, C)`. All entries satisfy $0 \le X_{ij} \le 127$.
- The function must return three arrays:
  1. `full_accum`: the per‑column sum using the full range ($q_{\max}=127$), stored as `int32`.
  2. `reduced_accum`: the per‑column sum after clamping each entry to $63$, also `int32`.
  3. `peak_per_col`: for each column, the maximum intermediate partial sum that would be seen when accumulating rows in order, using the reduced range. This array must be of type `int32`.

All computations should use NumPy only; no explicit Python loops.

## Example

```python
import numpy as np
X = np.array([[0, 127],
              [63, 64],
              [127, 0]], dtype=np.uint8)

full_accum, reduced_accum, peak_per_col = reduce_range_accumulator_safety(X)
print(full_accum)      # [190 191]
print(reduced_accum)   # [90  91]
print(peak_per_col)    # [90  91]  (no overflow for these small values)
```

## What the gate checks

Two metrics are evaluated:

1. **Relative error** – The global relative L2 error between each returned array and a NumPy oracle must satisfy  
   $\mathrm{rel\_err} \le 10^{-3}$.

2. **Peak safety** – Every element of `peak_per_col` must be less than or equal to the maximum signed 32‑bit integer, $2^{31}-1 = 2147483647$. The gate passes only if this condition holds for all columns.

The grader recomputes the reference values on the fly; no hard‑coded numbers are used. A correct implementation will satisfy both metrics. A broken starter that, for example, halves values instead of clamping or uses a 32‑bit sum that overflows will fail one or both gates.
