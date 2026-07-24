## Context

A linear layer stores its weights as $W \in \mathbb{R}^{C_{out} \times C_{in}}$, where row $i$ is the
weight vector of **output channel** $i$. Symmetric uniform quantization to $b$ bits maps a real
value to an integer code with a single scale factor:

$$ q = \mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{w}{s}\right), -q_{\max}, q_{\max}\right),
\qquad \hat{w} = q \cdot s, \qquad q_{\max} = 2^{\,b-1} - 1 . $$

**Granularity** decides how many scales you keep. *Per-tensor* uses one $s$ for the whole matrix.
*Per-channel* (the standard for weights) keeps one scale per output channel, obtained by reducing
the absolute value **along the input dimension**:

$$ s_i = \frac{\max_j |W_{ij}|}{q_{\max}}, \qquad s \in \mathbb{R}^{C_{out} \times 1}. $$

Reducing along the other axis silently produces $C_{in}$ scales of shape $1 \times C_{in}$. Because
NumPy happily broadcasts that against $W$, nothing crashes — the bug is invisible until you measure
quality. Rows whose magnitude is much smaller than the loudest row get divided by a step size set by
that loud row, round to $0$, and are effectively deleted. A global MSE hides this (those rows carry
little energy); a **per-channel** relative error, which weights every row equally, does not:

$$ \mathrm{channel\_rel\_err} = \frac{1}{C_{out}} \sum_{i}
\frac{\lVert \hat{W}_i - W_i \rVert_2}{\lVert W_i \rVert_2}. $$

## Task

`starter.py` contains a per-channel quantizer whose scale is reduced over the wrong axis. Fix it.

```python
def quantize_per_channel(W: np.ndarray, n_bits: int = 8) -> tuple[np.ndarray, np.ndarray]:
    ...
```

* `W` — float array of shape $(C_{out}, C_{in})$.
* `n_bits` — bit width $b$; the grader calls you at both $b = 8$ and $b = 4$, so the code range must
  actually be derived from `n_bits`.
* Returns `(q, scale)`:
  * `q` — `np.int8` array with the shape of `W` and values in $[-q_{\max}, q_{\max}]$,
  * `scale` — float array that broadcasts against `W` so that `q * scale` reconstructs $\hat{W}$;
    for per-output-channel quantization its shape is $(C_{out}, 1)$.

A row that is entirely zero must not divide by zero — use a scale of $1.0$ there.

## Example

```python
import numpy as np

W = np.array([[1.0, -0.5],
              [1e-3, 2e-3]])
q, scale = quantize_per_channel(W, n_bits=8)

# scale -> [[7.874e-03], [1.5748e-05]]   one scale per ROW
# q     -> [[127, -64], [64, 127]]       the quiet row survives
#
# With the scale reduced over axis 0 the second row would round to [0, 0].
```

## What the gate checks

The grader builds a deterministic weight matrix with `np.random.default_rng(0)` whose row magnitudes
span three orders of magnitude, calls your function, dequantizes with **your** returned `scale`, and
compares against the original using NumPy as the oracle. Nothing is hardcoded: the reference value is
recomputed at grade time and reported as the info metric `ref_channel_rel_err`.

* `channel_rel_err` $\le 0.01$ — mean per-row relative error at 8 bits (correct axis $\approx 0.0063$,
  wrong axis $\approx 0.26$).
* `channel_rel_err_int4` $\le 0.2$ — the same measurement at 4 bits (correct axis $\approx 0.12$),
  so `n_bits` must really be used.
* `codes_valid` $= 1$ — at both bit widths `q` has dtype `int8`, the shape of `W`, and
  $|q| \le q_{\max}$.
* `size_ratio` $\ge 3.5$ — computed from the real `nbytes` of what you return, so the codes must
  genuinely be `int8` and the scale must stay one value per channel.
