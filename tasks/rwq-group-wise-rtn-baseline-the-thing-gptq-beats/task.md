## Context

Round-to-nearest (RTN) quantization is the baseline every fancier method
(GPTQ, AWQ, HQQ, ...) is measured against: quantize each weight
independently, no error compensation, no calibration-aware reordering. The
one lever RTN does have is **granularity** — quantizing per-tensor is
cheap but crude; quantizing **per group of columns** (e.g. every 128
columns get their own scale) captures local magnitude variation at a small
metadata cost, and is what real int4 weight-only quantizers (e.g. GPTQ's
own baseline, bitsandbytes NF4 blocks) use in practice.

This task implements **per-row, per-group symmetric int4 RTN** — the exact
"no error feedback" baseline that GPTQ-style methods are designed to beat.

### Algorithm

Given $W \in \mathbb{R}^{d_{out}\times d_{in}}$ and a group size $G$
(with $G \mid d_{in}$), split each row into $d_{in}/G$ contiguous groups of
$G$ columns. For row $r$ and group $g$ (columns $gG,\dots,(g{+}1)G{-}1$):

$$
a_{r,g} = \max_{j \in \text{group } g} |W_{r,j}|, \qquad
s_{r,g} = \begin{cases} a_{r,g}/7 & a_{r,g} > 0 \\ 1 & a_{r,g}=0 \end{cases}
$$

$$
\text{code}_{r,j} = \mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{W_{r,j}}{s_{r,g}}\right),\, -7,\, 7\right)
\qquad (j \text{ in group } g)
$$

This is **symmetric int4**: 4-bit signed codes restricted to $[-7,7]$ (no
$-8$), scaled per row *and* per group — no zero-point, no error feedback
onto other columns.

Dequantization: $\hat W_{r,j} = \text{code}_{r,j}\cdot s_{r,g}$.

## Task

Implement `rtn_group_quantize`:

```python
def rtn_group_quantize(W: np.ndarray, group_size: int) -> tuple[np.ndarray, np.ndarray]:
    ...
```

* `W` — `(d_out, d_in)` weight matrix.
* `group_size` — group width $G$; you may assume `d_in % group_size == 0`.

Return `(codes, Wq)`:

* `codes` — integer array, shape `(d_out, d_in)`, values in `[-7, 7]`, the
  quantization codes above.
* `Wq` — float array, shape `(d_out, d_in)`, the dequantized reconstruction
  `codes * scale` (each element scaled by its own row's group scale).

## Example

```python
import numpy as np
W = np.array([[0.1, -6.0, 0.2, 5.9]])  # d_in=4, two groups of size 2
codes, Wq = rtn_group_quantize(W, group_size=2)
# group 0 = [0.1, -6.0]: amax=6.0, scale=6.0/7
# group 1 = [0.2, 5.9]:  amax=5.9, scale=5.9/7
```

## What the gate checks

* **codes_exact_match** — your integer `codes` array must exactly equal a
  NumPy oracle running the per-row, per-group absmax formula above on the
  fixed weight fixture (`gptq_w.npy`, group size 128).
* **recon_max_abs_err** — the max-abs difference between your `Wq` and the
  oracle's dequantized reconstruction on the same fixture.
