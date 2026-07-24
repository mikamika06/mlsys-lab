## Context

A linear layer computes $Y = XW^\top$ with weights $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$
and calibration activations $X \in \mathbb{R}^{n \times d_{\text{in}}}$.

**Round-to-nearest (RTN)** quantizes every weight independently. With symmetric $b$-bit
per-output-row scales,

$$
q_{\max} = 2^{b-1} - 1, \qquad
s_i = \frac{\max_j |W_{ij}|}{q_{\max}}, \qquad
\widehat W_{ij} = s_i \cdot \operatorname{clip}\!\left(\operatorname{round}\!\left(\frac{W_{ij}}{s_i}\right), -q_{\max}, q_{\max}\right).
$$

RTN minimises weight error, not *output* error — it ignores $X$ entirely.

**GPTQ** quantizes column by column and pushes the rounding residual of each column into the
not-yet-quantized columns, so that the layer output stays close to the original. It uses the
Hessian of the layer reconstruction loss,

$$
H = X^\top X + \lambda \, \overline{\operatorname{diag}(X^\top X)} \, I ,
\qquad \lambda = 0.01 ,
$$

where $\overline{\operatorname{diag}(\cdot)}$ is the mean of the diagonal (the standard dampening
term). Let $U$ be the **upper**-triangular Cholesky factor of the inverse Hessian,

$$
H^{-1} = U^\top U .
$$

Then, for columns $j = 0, 1, \dots, d_{\text{in}}-1$, with $w = W_{:,j}$ the *current* (already
compensated) column:

$$
q = s \cdot \operatorname{clip}\!\left(\operatorname{round}\!\left(\frac{w}{s}\right), -q_{\max}, q_{\max}\right),
\qquad
e = \frac{w - q}{U_{jj}},
$$

$$
W_{:,k} \leftarrow W_{:,k} - e \, U_{jk} \quad \text{for all } k > j .
$$

The scale vector $s$ is computed **once, from the original $W$**, and reused for every column.

## Task

Implement both quantizers in `solve.py`:

```python
def quantize_rtn(W: np.ndarray, bits: int) -> np.ndarray: ...
def quantize_gptq(W: np.ndarray, X: np.ndarray, bits: int, damp: float = 0.01) -> np.ndarray: ...
```

Both return the **de-quantized** weights: a float array of shape $(d_{\text{out}}, d_{\text{in}})$
holding $s_i \cdot k$ values on the quantization grid. Do not modify the input arrays in place.

Fixtures `W.npy` $(32 \times 48)$ and `X.npy` $(256 \times 48)$ hold one calibration pair; the grader
also runs a second, freshly generated pair.

## Example

```python
import numpy as np

W = np.array([[1.0, -0.5, 0.25],
              [0.4,  0.8, -0.2]])
X = np.random.default_rng(0).normal(size=(64, 3))

W_rtn  = quantize_rtn(W, 4)
W_gptq = quantize_gptq(W, X, 4)

# Same grid, different rounding decisions:
#   ||X @ W_gptq.T - X @ W.T||  <  ||X @ W_rtn.T - X @ W.T||
```

## What the gate checks

The grader recomputes both references with NumPy on the same inputs (nothing is hardcoded) and
reports, over all cases:

* `rtn_rel_err` — $\lVert \widehat W_{\text{RTN}} - \widehat W^{\text{ref}}_{\text{RTN}} \rVert / \lVert \widehat W^{\text{ref}}_{\text{RTN}} \rVert$, must be $\le 10^{-9}$;
* `gptq_rel_err` — the same relative error for the GPTQ weights, must be $\le 10^{-6}$;
* `gptq_over_rtn_output_error` — the ratio of **layer-output** relative errors

$$
\frac{\lVert X\widehat W_{\text{GPTQ}}^\top - XW^\top \rVert}{\lVert X\widehat W_{\text{RTN}}^\top - XW^\top \rVert} ,
$$

  computed from *your* returned weights, must be $\le 0.9$ — i.e. error compensation has to
  actually buy at least a 10% output-error reduction over RTN.

`rtn_output_rel_err` and `gptq_output_rel_err` are reported for information.
