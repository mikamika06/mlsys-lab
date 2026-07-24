## Context

The two common 8-bit float formats (OCP FP8) trade mantissa precision for
exponent range:

* **E4M3**: 1 sign, 4 exponent bits (bias $7$), 3 mantissa bits. Max finite
  magnitude $448$. Only the single code `(exponent=1111, mantissa=111)` is
  reserved (NaN) — every other exponent=`1111` code is a normal finite value
  ("FN" style).
* **E5M2**: 1 sign, 5 exponent bits (bias $15$), 2 mantissa bits. Max finite
  magnitude $57344$. The **entire** top exponent block (`exponent=11111`,
  any mantissa) is reserved for Inf/NaN (standard IEEE-style reservation) —
  more range, but only 2 mantissa bits of precision.

For exponent field $e$ and mantissa field $m$ (with `MBITS` mantissa bits),
a code decodes as:

$$
\text{value}(s,e,m) =
\begin{cases}
(-1)^s \cdot 2^{1-\text{bias}} \cdot \dfrac{m}{2^{\text{MBITS}}} & e = 0 \\[4pt]
(-1)^s \cdot 2^{\,e-\text{bias}} \cdot \left(1 + \dfrac{m}{2^{\text{MBITS}}}\right) & \text{otherwise, if not reserved}
\end{cases}
$$

For this task, **casting** a real value to a format means snapping it to the
**nearest finite representable value** of that format (never producing
Inf/NaN) — the standard behaviour of a storage-cast quantizer.

Whichever format gives the smaller reconstruction MSE on a given tensor
"wins" — a tensor of small, tightly-clustered values favors E4M3 (finer
mantissa grid near the values actually present); a tensor with rare huge
outliers favors E5M2 (its larger exponent range keeps the outliers from
saturating and dominating the error).

## Task

Implement `compare_fp8_formats`:

```python
def compare_fp8_formats(x: np.ndarray) -> tuple[float, float, str]:
    ...
```

* `x` — a float array of any shape.

Build both 256-entry grids (E4M3 and E5M2) as defined above, cast `x` to the
nearest finite value of each grid, and return
`(mse_e4m3, mse_e5m2, winner)`:

* `mse_e4m3`, `mse_e5m2` — `float`, the mean squared error
  $\frac{1}{n}\sum (\hat x_i - x_i)^2$ of casting `x` to that grid and back.
* `winner` — the string `"e4m3"` or `"e5m2"`, whichever has the smaller MSE
  (ties go to `"e4m3"`).

## Example

```python
import numpy as np
x = np.array([0.01, 0.02, -0.015, 100000.0])   # tiny cluster + one huge outlier
mse_e4m3, mse_e5m2, winner = compare_fp8_formats(x)
# E4M3 saturates the outlier hard (max 448) -> large MSE; E5M2's bigger range
# (max 57344) represents it far better -> winner == "e5m2"
```

## What the gate checks

* **mse** — max absolute difference between your `(mse_e4m3, mse_e5m2)` and
  the values from a NumPy oracle that builds both grids and casts
  identically, across several random tensors (some narrow-range, some with
  injected outliers).
* **exact_match** — your reported `winner` string must match the oracle's on
  every trial.
