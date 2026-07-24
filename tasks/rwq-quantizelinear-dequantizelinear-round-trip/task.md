## Context

ONNX's `QuantizeLinear` / `DequantizeLinear` op pair is the standard
representation of an asymmetric affine (`uint8`) quantized tensor in
inference graphs. Given a scalar `scale` $s > 0$ and integer
`zero_point` $z \in [0,255]$:

$$
\text{QuantizeLinear}(x) = q = \mathrm{clip}\big(\mathrm{round}(x/s) + z,\; 0,\; 255\big) \quad (\texttt{uint8})
$$

$$
\text{DequantizeLinear}(q) = \hat x = (q - z)\cdot s
$$

`round` here is round-half-to-even (ONNX's convention, and also
NumPy's default `np.round`/`np.rint` behavior — no special handling
needed).

## Task

Implement `qdq_round_trip`:

```python
def qdq_round_trip(x: np.ndarray, scale: float, zero_point: int):
    ...
```

- `x`: `float64` array, any shape.
- `scale`: positive Python `float`.
- `zero_point`: Python `int` in `[0, 255]`.

1. `q = clip(round(x / scale) + zero_point, 0, 255)`, cast to `uint8`.
2. `deq = (q.astype(float64) - zero_point) * scale`.

Return `(q, deq)`.

## Example

```python
import numpy as np
x = np.array([-1.0, 0.0, 0.5, 3.2])
q, deq = qdq_round_trip(x, scale=0.1, zero_point=128)
# q[1] = clip(round(0/0.1) + 128, 0, 255) = 128
# q[3] = clip(round(3.2/0.1) + 128, 0, 255) = clip(32 + 128, 0, 255) = 160
# deq[1] = (128 - 128) * 0.1 == 0.0
```

## What the gate checks

The grader builds several seeded `(x, scale, zero_point)` cases,
including values that saturate at both ends of `[0, 255]`, and applies
the exact QuantizeLinear/DequantizeLinear formulas independently in
NumPy.

`codes_exact_match` is `1.0` only if your `q` matches the oracle's
`uint8` codes exactly on every element of every case (integer
comparison, no tolerance) — this catches a wrong rounding mode, a
missing clip, or an off-by-one in the zero-point sign. `dequant_max_abs_err`
is the worst-case max elementwise absolute difference between your
`deq` and the oracle's (must be `<= 1e-6`) — this catches a correct
`q` paired with a wrong dequantization formula (e.g. `(q + z) * s`
instead of `(q - z) * s`).
