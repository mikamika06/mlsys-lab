## Context

Structured model architectures often have constraints that describe how many components survive after pruning or masking. A candidate binary mask can be accepted only when the number of active components matches the requested architecture.

Consider a transformer-like layout with $L$ layers, $H$ attention heads per layer, and feed-forward width $d_{ff}$. A candidate mask is a boolean array with shape $(L, H + d_{ff})$. For each layer $l$, the first $H$ entries represent attention heads and the remaining $d_{ff}$ entries represent feed-forward units.

The number of surviving heads and feed-forward units is computed by counting active mask entries:

$$
h_l = \sum_{i=1}^{H} m_{l,i}
$$

$$
f_l = \sum_{i=1}^{d_{ff}} m_{l,H+i}
$$

The architecture constraint is satisfied when every layer has exactly the requested number of active heads and feed-forward units.

## Task

Implement `classify_mask(mask, L, H, d_ff, target_heads, target_ff)`:

```python
def classify_mask(mask, L, H, d_ff, target_heads, target_ff):
    ...
```

The function receives:

- `mask`: a NumPy boolean array of shape $(L, H+d_{ff})$.
- `L`: number of layers.
- `H`: total attention heads per layer.
- `d_ff`: total feed-forward units per layer.
- `target_heads`: required active heads per layer.
- `target_ff`: required active feed-forward units per layer.

Return `True` only if every layer exactly matches both target counts. Otherwise return `False`.

Use NumPy operations to count survivors. Do not inspect individual entries with nested Python loops.

## Example

```python
import numpy as np

mask = np.array([
    [1, 1, 0, 1, 0, 0],
    [1, 0, 1, 1, 0, 0],
], dtype=bool)

# L=2, H=3, d_ff=3
# Each layer has 2 active heads and 1 active FF unit.
ok = classify_mask(mask, 2, 3, 3, 2, 1)
# True
```

## What the gate checks

The gate generates several masks with different layer, head, and feed-forward constraints. It computes the expected answer by using a NumPy oracle that sums survivors along each architecture axis.

The `exact_match` score must be $1.0`. A result is accepted only when the returned boolean matches the oracle for every generated case.
