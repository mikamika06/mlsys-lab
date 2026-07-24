## Context

Structured pruning methods often learn continuous gate values and later convert them into a smaller target architecture. A gate value represents how strongly a structure should be kept. Hard-concrete pruning commonly converts learned probabilities into binary decisions, but a deployment target may require an exact number of layers, attention heads, or feed-forward dimensions.

For a collection of scores $s_1, s_2, \dots, s_n$, selecting the largest $k$ structures means choosing the index set

$$
K = \operatorname{arg\,top}_k(s) .
$$

The target architecture specifies exact counts $(L_t, H_t, d_t)$ for layers, heads, and feed-forward dimensions. The mapping procedure first selects the best $L_t$ layers, then selects the best $H_t$ heads inside every selected layer, and finally selects the best $d_t$ feed-forward dimensions.

## Task

Implement `map_mask_to_arch`:

```python
def map_mask_to_arch(layer_gates, head_gates, dim_gates, target):
    ...
```

Inputs:

- `layer_gates`: a one-dimensional NumPy array of length $L$ containing layer scores.
- `head_gates`: a two-dimensional NumPy array of shape $(L, H)$ containing attention head scores.
- `dim_gates`: a one-dimensional NumPy array of length $d_{ff}$ containing feed-forward dimension scores.
- `target`: a tuple `(target_L, target_H, target_d_ff)` describing the required architecture size.

Return a dictionary:

```python
{
    "layers": [...],
    "heads": [[...], [...], ...],
    "dims": [...]
}
```

where all indices refer to the original unpruned architecture.

Selection rules:

1. Keep the `target_L` layers with the highest `layer_gates`.
2. For each kept layer, keep the `target_H` heads with the highest values from that layer of `head_gates`.
3. Keep the `target_d_ff` dimensions with the highest values in `dim_gates`.

When scores tie, choose the smaller index first. Returned indices must be sorted in ascending order.

## Example

```python
import numpy as np

layer_gates = np.array([0.2, 0.9, 0.5])
head_gates = np.array([
    [0.1, 0.8],
    [0.7, 0.3],
    [0.9, 0.4],
])
dim_gates = np.array([0.5, 0.1, 0.8])
target = (2, 1, 2)

result = map_mask_to_arch(
    layer_gates,
    head_gates,
    dim_gates,
    target,
)

# {
#   "layers": [1, 2],
#   "heads": [[0], [0]],
#   "dims": [0, 2]
# }
```

## What the gate checks

The gate computes an independent NumPy oracle that performs the same top-score selection with deterministic tie handling. The returned layer indices, per-layer head indices, and dimension indices must exactly match the oracle output.
