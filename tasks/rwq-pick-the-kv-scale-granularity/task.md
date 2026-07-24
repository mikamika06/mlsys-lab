## Context

KV cache quantization reduces memory usage by storing keys and values with lower precision. A scale determines how floating point values are mapped into an integer representation.

For a tensor $X$, symmetric int8 quantization with scale $s$ is

$$
\hat{X} = \mathrm{clip}\left(\mathrm{round}\left(\frac{X}{s}\right), -127, 127\right) s ,
$$

where the scale is chosen from the maximum absolute value of the values being quantized:

$$
s = \frac{\max(|X|)}{127}.
$$

KV caches can store scales at different granularities:

- per-tensor: one scale for the whole KV cache,
- per-token: one scale for each token position,
- per-head: one scale for each attention head.

Finer granularity usually lowers reconstruction error but requires more scale values. The selection problem is to minimize a combined objective:

$$
\mathrm{cost} = \mathrm{MSE}(O, \hat{O}) + \lambda \cdot \max(0, S - B),
$$

where $O$ is the original attention output, $\hat{O}$ is the output using quantized keys and values, $S$ is the number of stored scale bytes, $B$ is the scale storage budget, and $\lambda$ is a fixed penalty factor.

## Task

Implement `choose_kv_scale_granularity(K, V, Q, budget)`:

```python
def choose_kv_scale_granularity(
    K: np.ndarray,
    V: np.ndarray,
    Q: np.ndarray,
    budget: int,
) -> int:
    ...
```

Inputs have shape $(h, t, d)$ for keys and values, where $h$ is the number of
attention heads, $t$ is the number of cached tokens, and $d$ is the head
dimension. `Q` has shape $(h, 1, d)$.

Return the index of the best granularity:

- `0` for per-tensor scales,
- `1` for per-token scales,
- `2` for per-head scales.

Use NumPy operations only. The function must evaluate all three choices using
the same quantization and attention computation described in the context, then
return the index with the smallest objective.

## Example

```python
import numpy as np

K = np.array([[[1.0, 2.0], [3.0, 4.0]]])
V = np.array([[[1.0, 0.0], [0.0, 2.0]]])
Q = np.array([[[1.0, 1.0]]])

choice = choose_kv_scale_granularity(K, V, Q, budget=64)
# choice is 0, 1, or 2
```

## What the gate checks

The gate builds deterministic KV cache examples and computes the oracle objective
for all three granularities. The returned index must exactly match the NumPy
oracle's argmin. The objective comparison uses deterministic float64 arithmetic
and the selected result must match within the gate tolerance of $10^{-6}$.
