## Context

Given a linear layer with weight matrix $W \in \mathbb{R}^{d_{out}\times d_{in}}$
and a batch of calibration activations $X \in \mathbb{R}^{n\times d_{in}}$, we
want to quantize $W$ to a low bit-width while keeping the layer's output
$Y = XW^\top$ as close as possible to the full-precision output.

**RTN (round-to-nearest)** quantizes every row of $W$ independently, with no
knowledge of $X$. For output channel $r$, with symmetric grid of `bits`
levels:

$$
s_r = \frac{\max_j |W_{r,j}|}{2^{bits-1}-1}, \qquad
\hat W_{r,j} = s_r \cdot \mathrm{clip}\!\Big(\mathrm{round}\big(\tfrac{W_{r,j}}{s_r}\big),\, -(2^{bits-1}-1),\, 2^{bits-1}-1\Big).
$$

**GPTQ** uses the *same* per-row grid $s_r$ (so both methods quantize to
identical levels) but processes columns $j = 0, 1, \dots, d_{in}-1$ in order
and, after rounding column $j$, feeds the rounding error back into the
*not-yet-quantized* columns, weighted by the inverse of the input Hessian
$H = X^\top X / n$. Concretely, with damping $\lambda = 0.01$:

$$
H \leftarrow \frac{X^\top X}{n} + \lambda\,\overline{\mathrm{diag}(H)}\, I, \qquad
H^{-1} = \text{Hinv (updated in place below)}.
$$

For $j = 0, \dots, d_{in}-1$ (with $W$ replaced in place by the running
error-fed working copy):

$$
d = Hinv_{jj}, \qquad
q = \text{RTN-round column } j \text{ of } W \text{ using } s_r,\qquad
e = \frac{W_{:,j} - q}{d}
$$
$$
W_{:,j+1:} \mathrel{-}= e \, Hinv_{j,\,j+1:}, \qquad
Hinv_{j+1:,\,j+1:} \mathrel{-}= \frac{Hinv_{j+1:,j}\,Hinv_{j,j+1:}}{d}.
$$

This is the standard (non-blocked, fixed left-to-right order) GPTQ update:
each rounding error is redistributed to the remaining columns in proportion
to how strongly they are coupled to column $j$ through the calibration
Hessian, which is exactly why it achieves lower output error than RTN at the
same bit-width.

## Task

Implement `gptq_vs_rtn_output_error(W, X, bits)`:

```python
def gptq_vs_rtn_output_error(W: np.ndarray, X: np.ndarray, bits: int) -> tuple[float, float]:
    ...
```

- `W`: `(d_out, d_in)` float64 weight matrix.
- `X`: `(n, d_in)` float64 calibration activations.
- `bits`: int bit-width (e.g. `4`).

Quantize `W` with RTN and with GPTQ as defined above (both using the same
per-row symmetric grid), then return `(mse_rtn, mse_gptq)` where

$$
\mathrm{mse} = \frac{1}{n\,d_{out}}\big\lVert XW^\top - X\hat W^\top\big\rVert_F^2 .
$$

## Example

```python
import numpy as np
rng = np.random.default_rng(1)
X = rng.standard_normal((64, 8))
W = rng.standard_normal((6, 8))
mse_rtn, mse_gptq = gptq_vs_rtn_output_error(W, X, bits=4)
mse_gptq < mse_rtn   # True — GPTQ's Hessian-aware error feedback wins
```

## What the gate checks

The grader builds its own RTN and GPTQ oracle (same formulas above, `numpy`
only, `np.linalg.inv` for $H^{-1}$, damping $\lambda=0.01$) over 5
deterministic calibration problems (`np.random.default_rng(0)`, correlated
`X` columns so the Hessian is non-diagonal and GPTQ's cross-column feedback
actually matters, `d_in=8`, `d_out=6`, `bits=4`). Three metrics are graded:

- `rtn_mse_error` / `gptq_mse_error`: absolute difference between your
  returned MSE and the oracle's, summed to the worst single-trial deviation;
  gate `<= 1e-6`.
- `gptq_beats_rtn`: `1.0` if your total GPTQ MSE (summed over the 5 trials)
  is strictly less than your total RTN MSE, else `0.0`; gate `== 1.0`. A
  correct GPTQ implementation always wins on these correlated fixtures — an
  implementation that just calls RTN twice, gets the Hessian update sign
  wrong, or forgets the error feedback will fail this gate even if its RTN
  number alone is correct.
