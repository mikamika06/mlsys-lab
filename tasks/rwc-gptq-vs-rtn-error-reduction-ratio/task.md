## Context

A linear layer computes $Y = XW^\top$ with weights $W \in
\mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ and calibration
activations $X \in \mathbb{R}^{n \times d_{\text{in}}}$.

**Round-to-nearest (RTN)** quantizes every weight independently, with
symmetric $b$-bit per-output-row scales:

$$
q_{\max} = 2^{b-1} - 1, \qquad
s_i = \frac{\max_j |W_{ij}|}{q_{\max}}, \qquad
\widehat W^{\text{RTN}}_{ij} = s_i \cdot \operatorname{clip}\!\left(\operatorname{round}\!\left(\frac{W_{ij}}{s_i}\right), -q_{\max}, q_{\max}\right).
$$

RTN minimizes weight error, ignoring $X$ (and therefore the actual layer
output) entirely.

**GPTQ** quantizes column by column and pushes each column's rounding
residual into the not-yet-quantized columns, keeping the layer's *output*
close to the original. It uses the (dampened) Hessian of the layer
reconstruction loss,

$$
H = X^\top X + \lambda \, \overline{\operatorname{diag}(X^\top X)} \, I, \qquad \lambda = 0.01,
$$

where $\overline{\operatorname{diag}(\cdot)}$ is the mean of the diagonal. Let
$U$ be the **upper**-triangular Cholesky factor of the inverse Hessian,
$H^{-1} = U^\top U$. Then for columns $j = 0, \dots, d_{\text{in}}-1$, with
$w = W_{:,j}$ the *current* (already-compensated) column, using the
**same** row scales $s$ computed once from the original $W$:

$$
q = s \cdot \operatorname{clip}\!\left(\operatorname{round}\!\left(\frac{w}{s}\right), -q_{\max}, q_{\max}\right),
\qquad
e = \frac{w - q}{U_{jj}},
\qquad
W_{:,k} \leftarrow W_{:,k} - e\, U_{jk} \quad \text{for all } k > j .
$$

The point of GPTQ's error compensation is that it reduces **layer-output**
error, not just weight error. That benefit can be quantified directly as
a ratio of output reconstruction errors on the same weight and the same
calibration activations:

$$
\rho = \frac{\lVert X\widehat W^{\text{GPTQ}\top} - XW^\top \rVert_F}{\lVert X\widehat W^{\text{RTN}\top} - XW^\top \rVert_F} .
$$

$\rho < 1$ means GPTQ's compensation actually reduced output error versus
plain rounding on the same weight.

## Task

Implement `gptq_vs_rtn_error_ratio(W, X, bits)`:

```python
def gptq_vs_rtn_error_ratio(W, X, bits):
    ...
```

- `W`: NumPy array of shape $(d_{\text{out}}, d_{\text{in}})$.
- `X`: NumPy array of shape $(n, d_{\text{in}})$.
- `bits`: positive int, the quantization bit-width $b$.

1. Compute $\widehat W^{\text{RTN}}$ (round-to-nearest) as defined above.
2. Compute $\widehat W^{\text{GPTQ}}$ (the full column-by-column
   Hessian-compensated procedure) with dampening $\lambda = 0.01$, as
   defined above.
3. Return the scalar ratio $\rho$ (a Python `float`) of layer-output
   Frobenius-norm reconstruction errors, as defined above.

## Example

```python
import numpy as np

W = np.load("fixtures/w.npy")
X = np.load("fixtures/x.npy")

rho = gptq_vs_rtn_error_ratio(W, X, bits=4)
# rho < 1.0 -- GPTQ's error compensation reduces output error relative
# to plain round-to-nearest on the same weight and activations.
```

## What the gate checks

The gate loads the committed `w.npy`/`x.npy` calibration fixture and
recomputes both quantizers and the ratio $\rho$ itself, exactly as
specified above (nothing hardcoded — everything derives from the actual
`W`/`X` values). Your reported ratio is compared against the oracle's
with relative error, threshold $10^{-6}$. Reporting the weight-error
ratio instead of the layer-output-error ratio, dropping the Hessian
dampening term, using per-column instead of frozen-from-the-original-$W$
scales, or getting the GPTQ column-update sign wrong will all produce a
different ratio than the oracle's.
