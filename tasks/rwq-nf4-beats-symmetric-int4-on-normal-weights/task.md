## Context

Pretrained neural network weights are, block by block, close to
zero-mean Gaussian. A **uniform** INT4 grid (equally spaced levels)
wastes half its 16 codes on the tails of the distribution, where almost
no probability mass lives, and starves the dense region around zero
where nearly all the values actually are. **NF4** ("NormalFloat4"),
introduced by QLoRA and used by `bitsandbytes` for 4-bit weight storage,
instead uses 16 **fixed, non-uniform** levels placed at the quantiles of
a standard normal distribution — so each of the 16 codes covers an
equal-probability slice of a Gaussian, packing precision where the
weights actually live. On real (near-Gaussian) weight blocks this gives
NF4 a strictly lower reconstruction MSE than a same-bit-width symmetric
uniform INT4 grid.

### The two quantizers

**NF4** — with the fixed codebook $C \in \mathbb{R}^{16}$ below (already
sorted, already the values `bitsandbytes` ships):

$$
C = [-1,\ -0.6961928,\ -0.5250731,\ -0.3949175,\ -0.2844414,\ -0.1847734,
\ -0.0910500,\ 0,$$
$$0.0795803,\ 0.1609302,\ 0.2461123,\ 0.3379152,\ 0.4407098,\ 0.5626170,
\ 0.7229568,\ 1]
$$

for a block $w$, let $a = \max_i |w_i|$ (absmax). Normalize
$\tilde w_i = w_i / a$, snap each to its nearest codebook level
$c^\star_i = \arg\min_{c \in C} |\tilde w_i - c|$, and dequantize
$\widehat w_i = c^\star_i \cdot a$.

**Symmetric INT4** — same absmax $a$, uniform scale $s = a / 7$, codes
$q_i = \mathrm{clip}(\mathrm{round}(w_i / s),\, -8,\, 7)$, dequantize
$\widehat w_i = q_i \cdot s$.

Both report reconstruction MSE: $\frac{1}{n}\sum_i (w_i - \widehat w_i)^2$.

## Task

Implement:

```python
def nf4_vs_int4_mse(w: list[float]) -> tuple[float, float]:
    ...
```

* `w` — 1-D array of weights (one block), roughly Gaussian-distributed.
* Return `(mse_nf4, mse_int4)`: the reconstruction MSE of `w` under each
  scheme above, using the fixed codebook $C$ given in Context for NF4.

## Example

```python
rng = random.Random(1)
w = rng.normal(size=256)
mse_nf4, mse_int4 = nf4_vs_int4_mse(w)
assert mse_nf4 < mse_int4   # NF4's quantile-spaced levels fit the Gaussian bulk better
```

## What the gate checks

* **max_abs_err** — both `mse_nf4` and `mse_int4` must match a Python
  oracle implementing the exact formulas above (same codebook, same
  absmax/scale) to within $10^{-9}$ absolute error, on several random
  Gaussian-ish blocks of length 256 (fixed seed).
* **ordering_ok** — on every one of those blocks, your returned
  `mse_nf4` must be strictly less than your `mse_int4` (a solution that
  swaps the two quantizers, or implements one of them wrong, will
  usually violate this even if the numbers otherwise look plausible).
