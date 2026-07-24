## Context

The "fast inverse square root" trick (famous from Quake III Arena) computes an
approximation of $1/\sqrt{x}$ for a positive 32-bit float $x$ without dividing
or taking a square root. It reinterprets the IEEE-754 bit pattern of $x$ as an
integer, does one integer subtraction against a magic constant, then
reinterprets the result back as a float:

$$
i = \mathrm{bits}(x) \qquad
j = \mathtt{0x5f3759df} - (i \gg 1) \qquad
y_0 = \mathrm{bits}^{-1}(j)
$$

Here $\mathrm{bits}(\cdot)$ is the raw 32-bit reinterpretation of a `float32`
as a `uint32` (no numeric conversion — a bit-for-bit view). Halving the
integer bit pattern approximately halves the float's base-2 exponent, and the
magic constant corrects the resulting bias so that $y_0 \approx x^{-1/2}$.

$y_0$ alone is a coarse approximation (worst-case relative error a few
percent). One iteration of Newton–Raphson on $f(y) = 1/y^2 - x$ sharpens it
considerably:

$$
y_1 = y_0 \left( \frac{3}{2} - \frac{x}{2} y_0^2 \right)
$$

## Task

Implement two functions:

```python
def rsqrt_raw(x: np.ndarray) -> np.ndarray:
    ...

def rsqrt_newton(x: np.ndarray) -> np.ndarray:
    ...
```

- `x` is a NumPy array of positive `float32` values (any shape).
- `rsqrt_raw` must return the magic-constant bit-hack approximation $y_0$
  described above, computed with the **exact** constant `0x5f3759df`, as a
  `float32` array with the same shape as `x`. It must be obtained via an
  integer bit-view of `x` (`x.view(np.uint32)` or equivalent) — not via
  `1/np.sqrt(x)` or any other numerically-exact route.
- `rsqrt_newton` must return $y_1$: `rsqrt_raw(x)` refined by exactly **one**
  Newton–Raphson step, as a `float32` array with the same shape as `x`.

## Example

```python
import numpy as np

x = np.array([1.0, 4.0, 100.0], dtype=np.float32)

y0 = rsqrt_raw(x)      # coarse: e.g. [0.99763..., 0.49881..., 0.09973...]
y1 = rsqrt_newton(x)   # refined: e.g. [0.99826..., 0.49957..., 0.09998...]

# true 1/sqrt(x) = [1.0, 0.5, 0.1]
```

## What the gate checks

The grader builds an oracle `1/sqrt(x)` in `float64` over a wide log-uniform
range of positive magnitudes (plus exact powers of two and a few round
numbers), and separately computes the bit-hack's expected raw output directly
from a real `uint32` view of `x` (the same operation your code must perform).

Three gates apply:

- **`raw_bit_exact`** — `rsqrt_raw(x)` must match the oracle's bit-hack output
  **exactly**, byte for byte (`byte_exact_fraction == 1.0`). This forces the
  exact magic constant and the exact shift-and-subtract sequence — any
  numerically-exact substitute (e.g. calling `np.sqrt`) will not reproduce
  these bit patterns.
- **`newton_rel_err`** — the global relative $L_2$ error of `rsqrt_newton(x)`
  against the `float64` oracle,
  $$
  \mathrm{rel\_err} = \frac{\lVert y_1 - x^{-1/2} \rVert_2}{\lVert x^{-1/2} \rVert_2},
  $$
  must be $\le 5\times 10^{-3}$. Skipping the Newton step (returning $y_0$
  again) fails this gate by roughly an order of magnitude.
- **`newton_dtype_ok`** — `rsqrt_newton(x)` must be returned as `float32`.
