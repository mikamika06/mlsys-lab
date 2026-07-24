## Context

8-bit optimizer states (Adam moments, etc.) aren't quantized to a plain
uniform int8 grid — magnitudes there span many orders of magnitude, and a
uniform grid wastes almost all its levels on the rare large values while
crushing the common small ones. Production 8-bit optimizer libraries
instead use a **dynamic (exponent-fraction) map**: the 8-bit code is
split into a sign bit, a unary-coded exponent field, and a fraction field
whose *width shrinks as the exponent grows* — so small magnitudes (near
zero) get fine resolution and large magnitudes get coarse resolution,
mirroring how floating point already trades range for precision, but
fully enumerable up front as a fixed 256-entry lookup table (no runtime
float decoding needed).

For `total_bits = 8`, `signed = True`, `max_exponent_bits = 7`, and each
exponent index $i \in \{0, \dots, 6\}$:

$$
\text{fraction\_items}(i) = 2^i + 1, \qquad
\text{boundaries}(i) = \operatorname{linspace}(0.1,\ 1.0,\ \text{fraction\_items}(i))
$$

$$
\text{means}(i)_j = \frac{\text{boundaries}(i)_j + \text{boundaries}(i)_{j+1}}{2}, \quad j = 0, \dots, 2^i - 1
$$

$$
\text{scale}(i) = 10^{-(6) + i}
$$

Each exponent index $i$ contributes $2^i$ **positive** levels
$\text{scale}(i) \cdot \text{means}(i)$ and (since `signed=True`) $2^i$
**negative** levels $-\text{scale}(i) \cdot \text{means}(i)$.

## Task

Implement `create_dynamic_map`:

```python
def create_dynamic_map(signed: bool = True, max_exponent_bits: int = 7, total_bits: int = 8) -> np.ndarray:
    ...
```

For each $i$ from $0$ to `max_exponent_bits - 1` (in order), compute
`fraction_items`, `boundaries`, `means`, and `scale` exactly as above,
and append `scale * means` then `-scale * means` to a running list.

After the loop, append the two literal values `0.0` and `1.0` to the
list.

Sort the complete list ascending and return it as a 1-D `float64` NumPy
array. With the defaults, the loop contributes
$2 \cdot \sum_{i=0}^{6} 2^i = 2 \cdot 127 = 254$ values, plus the two
appended boundary values, for exactly `2 ** total_bits = 256` levels
total.

## Example

```python
import numpy as np

levels = create_dynamic_map()
levels.shape        # (256,)
levels.min(), levels.max()   # (~-0.993, 1.0)
np.any(levels == 0.0)        # True -- the exact zero entry
np.all(np.diff(levels) >= 0) # True -- sorted ascending
# small |i| -> scale ~1e-6, giving very fine resolution near 0;
# i = 6 -> scale = 1, giving the coarsest grid near +-1
```

## What the gate checks

The grader computes the reference 256-level array independently in
NumPy using the exact same `fraction_items` / `boundaries` / `means` /
`scale` construction above (defaults `signed=True`,
`max_exponent_bits=7`, `total_bits=8`), never calling your function.

`size_ok` requires your returned array to have the right shape (256,
matching `2 ** total_bits`), be sorted ascending, and contain an exact
`0.0` entry (must be `>= 1.0`) — this catches a wrong bit-budget count
or a missing/misplaced zero level before comparing values.
`max_abs_err` is the max elementwise absolute difference between your
sorted array and the oracle's (must be `<= 1e-4`) — this catches a wrong
`scale` exponent, wrong `fraction_items` formula, or a dropped
positive/negative branch, any of which shifts levels far outside
floating-point tolerance.
