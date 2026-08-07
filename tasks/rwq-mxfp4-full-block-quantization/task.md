## Context

OCP's **MXFP4** format quantizes weights in contiguous blocks of **32**
elements. Each block shares one power-of-two scale stored as **E8M0** (an
8-bit unsigned exponent, no mantissa, no sign — so the scale itself is
always an exact power of two), and every element inside the block is
individually rounded onto the **E2M1** grid — one sign bit, two exponent
bits, one mantissa bit, giving the 8-level magnitude set

$$
\mathcal{G} = \{0,\ 0.5,\ 1,\ 1.5,\ 2,\ 3,\ 4,\ 6\}.
$$

For a block $w = (w_1,\dots,w_{32})$, let $a=\max_i |w_i|$. The shared
exponent is the smallest non-negative integer that brings every scaled
element inside E2M1's range $[-6,6]$:

$$
e = \max\!\left(0,\ \left\lceil \log_2\!\left(\frac{a}{6}\right) \right\rceil\right)
\qquad\text{(}a=0 \Rightarrow e = 0\text{)}, \qquad
\text{scale} = 2^{e}.
$$

Each element is scaled down by that block's `scale`, then snapped to the
nearest signed value of $\mathcal{G}$ (ties broken toward the
lower-magnitude grid point, i.e. whichever comes first when scanning
$\mathcal{G}$ in increasing order):

$$
\mathrm{code}_i = \operatorname{sign}(w_i)\cdot \operatorname*{arg\,min}_{g\in\mathcal{G}}\left|\frac{|w_i|}{\text{scale}} - g\right|,
\qquad
\hat w_i = \mathrm{code}_i \cdot \text{scale}.
$$

Because every element in a block is divided by the *same* scale, a single
large-magnitude outlier forces the whole block's exponent up, coarsening
every other (possibly tiny) element in that block — this is the central
accuracy/hardware-efficiency tradeoff of shared microscaling formats.

## Task

Implement `mxfp4_full_block_quantize(W)`:

```python
def mxfp4_full_block_quantize(W: list[list[float]]) -> dict:
    ...
```

`W` is a list of shape $(B, 32)$: $B$ independent blocks of 32
values. Return a dict with three list:

- `"scale"`: shape $(B,)$, float64 — each block's power-of-two scale
  $2^{e}$ as defined above.
- `"codes"`: shape $(B, 32)$, float64 — each element's E2M1 grid value
  (one of $\{0,\pm0.5,\pm1,\pm1.5,\pm2,\pm3,\pm4,\pm6\}$), **before**
  scaling back up.
- `"dequant"`: shape $(B, 32)$, float64 — `codes * scale[:, None]`, the
  full round-trip reconstruction of `W`.

## Example

```python
W = [[0.0] * 32 for _ in range(1)]
W[0, 0] = 1.2
W[0, 1] = 12.0          # forces a=12 -> e = ceil(log2(12/6)) = 1 -> scale=2
out = mxfp4_full_block_quantize(W)
out["scale"][0]         # 2.0
out["codes"][0, 1]      # 6.0    (12/2 = 6, exactly on the grid)
out["dequant"][0, 1]    # 12.0   (6.0 * 2.0)
out["codes"][0, 0]      # 0.5    (1.2/2 = 0.6, nearest grid point is 0.5)
out["dequant"][0, 0]    # 1.0    (0.5 * 2.0)
```

## What the gate checks

The grader loads a committed fixture `mx_w.npy` (48 blocks of 32
"weight"-like values with widely varying per-block magnitude, including an
all-zero block, an exact power-of-two boundary case, and a block with a
single dominant outlier next to near-zero values) and computes the
reference `scale`, `codes`, `dequant` with an independent Python oracle
using the exact formulas above. Three metrics are reported, each the
maximum absolute element-wise difference against the oracle:

- `scale_err` — gate `< 1e-9` (scales are exact powers of two, so a
  correct implementation matches bit-for-bit).
- `code_err` — gate `< 1e-9` (codes live on the exact dyadic grid
  $\mathcal{G}$, again no meaningful tolerance — a wrong exponent or a
  wrong nearest-grid choice lands on a *different* grid point, not a
  close one).
- `dequant_err` — gate `< 1e-6` (the full round-trip reconstruction).

Getting the exponent formula wrong (e.g. omitting the `max(0, ...)`
clamp, mishandling the all-zero block, or an off-by-one in the ceiling)
or breaking the E2M1 grid lookup will fail one or more of these gates.
