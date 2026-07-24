## Context

Two real compression modifiers are frequently composed in production recipes:

1. **SparseGPT 2:4** — a Hessian-aware *structured* pruning scheme. For every
   contiguous group of 4 columns in a row, it keeps the 2 weights with the
   highest saliency and zeroes the other 2, then *compensates* the surviving
   weights in that row using the inverse of a calibration Hessian so the
   layer's output changes as little as possible.
2. **GPTQ** — a Hessian-ordered *quantization* scheme. It quantizes one
   weight column at a time to a low bit-width and immediately propagates the
   rounding error into the not-yet-quantized columns of the same row via the
   inverse Hessian, so later columns compensate for earlier rounding error.

A realistic mini-recipe runs SparseGPT 2:4 first, then runs GPTQ on the
*survivors* (the already-pruned matrix) to pack the remaining weights to
`W4A16` (4-bit weights, activations left at full precision).

Let $W \in \mathbb{R}^{m\times n}$ be a weight matrix ($n$ divisible by 4)
and $X \in \mathbb{R}^{n\times s}$ a calibration-activation matrix (columns
are samples, rows are input features).

**Stage 1 — SparseGPT 2:4 pruning.**
Form the damped Hessian

$$
H_p = 2\,X X^\top + \lambda_p I \in \mathbb{R}^{n\times n}, \qquad H_p^{-1} = H_p^{-1}.
$$

For every row $r$ and every group of 4 consecutive columns
$\{c_1,c_2,c_3,c_4\}$, score each column by

$$
\text{score}(c) = \frac{W[r,c]^2}{H_p^{-1}[c,c]},
$$

keep the 2 columns with the highest score, and prune the other 2. For each
pruned column $c$ (using its **pre-pruning** value $w_c$), compensate every
still-kept column $k$ in the same group of 4:

$$
W[r,k] \mathrel{-}= w_c \cdot \frac{H_p^{-1}[k,c]}{H_p^{-1}[c,c]}, \qquad W[r,c] = 0.
$$

**Stage 2 — GPTQ int-`bits` quantization of the survivors.**
Form a second damped Hessian from the same calibration activations,

$$
H_q = X X^\top + \delta\cdot\mathrm{mean}(\mathrm{diag}(X X^\top))\, I, \qquad H_q^{-1} = H_q^{-1}.
$$

Use a symmetric per-row scale $s_r = \max_j |W[r,j]| / q_{\max}$ with
$q_{\max} = 2^{\text{bits}-1}-1$. Process columns $i = 0,\dots,n-1$ in order:

$$
\hat w_i = \mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{W[:,i]}{s}\right), -q_{\max}, q_{\max}\right)\cdot s,
\qquad
e = \hat w_i - W[:,i],
$$

$$
W[:,i+1:] \mathrel{-}= e \otimes \frac{H_q^{-1}[i,\,i+1:]}{H_q^{-1}[i,i]}.
$$

Because the pruned entries are exactly `0`, they quantize back to exactly
`0` under this scheme (round(0) = 0), so the 2:4 sparsity pattern is
preserved through quantization automatically.

## Task

Implement `sparsegpt_then_gptq`:

```python
def sparsegpt_then_gptq(
    W: np.ndarray,
    X: np.ndarray,
    bits: int = 4,
    lam_prune: float = 1e-2,
    damp: float = 1e-2,
) -> np.ndarray:
    ...
```

* `W` — 2-D array of shape $(m, n)$, `float64`, $n$ divisible by 4.
* `X` — 2-D array of shape $(n, s)$, `float64` calibration activations.
* `bits` — target weight bit-width for the GPTQ stage (default `4`).
* `lam_prune` — Hessian damping used in the SparseGPT stage.
* `damp` — relative Hessian damping used in the GPTQ stage.

Return `W_hat`, the $(m, n)$ array obtained by running SparseGPT 2:4
pruning followed by GPTQ quantization on the survivors, exactly as
described above. Run the two stages in this order — pruning before
quantization — and use `float64` throughout.

## Example

```python
import numpy as np

rng = np.random.default_rng(0)
W = rng.standard_normal((4, 8))
X = rng.standard_normal((8, 16))

W_hat = sparsegpt_then_gptq(W, X)
print(W_hat.shape)                 # (4, 8)
print(np.mean(W_hat == 0.0))       # close to 0.5 (2:4 structured sparsity)
```

## What the gate checks

The grader builds several deterministic `(W, X)` calibration pairs, runs a
NumPy oracle that performs the same two-stage recipe (SparseGPT 2:4, then
GPTQ), and compares it against your output:

* **rel_err** — Frobenius relative error between your reconstructed layer
  output $\hat W X$ and the oracle's $\hat W_{\text{ref}} X$, must be
  $\le 10^{-5}$.
* **w_rel_err** — Frobenius relative error between your returned $\hat W$
  and the oracle's $\hat W_{\text{ref}}$ directly, must be $\le 10^{-5}$.

Both gates require your implementation to reproduce the exact same pruning
saliency, compensation, quantization order, and error propagation as the
reference recipe — not merely a similarly-sized error.
