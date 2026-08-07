## Context

QLoRA freezes a 4-bit NF4-quantized base weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ and trains a small full-precision low-rank adapter $B A$ on top of it. A forward pass through such a linear layer has to:

1. **Dequantize the base weight.** Each entry $W_{o,j}$ is stored as a 4-bit
   code $c_{o,j} \in \{0, \dots, 15\}$ indexing into the fixed 16-value NF4
   lookup table $L$, together with a per-block absmax scale. Columns are
   split into contiguous blocks of `blocksize` values; block
   $b = \lfloor j / \text{blocksize} \rfloor$ of row $o$ shares one scale
   $s_{o,b}$:
   $$
   \hat W_{o,j} = L[c_{o,j}] \cdot s_{o,\, \lfloor j / \text{blocksize} \rfloor}
   $$

2. **Add the LoRA delta.** With adapter matrices $A \in \mathbb{R}^{r \times d_{\text{in}}}$, $B \in \mathbb{R}^{d_{\text{out}} \times r}$ and scalar $\alpha$, the effective weight is
   $$
   W_{\text{eff}} = \hat W + \frac{\alpha}{r} \, B A
   $$

3. **Run the linear layer.** For input $x \in \mathbb{R}^{n \times d_{\text{in}}}$,
   $$
   y = x \, W_{\text{eff}}^\top \in \mathbb{R}^{n \times d_{\text{out}}}
   $$

The 16 NF4 quantization levels (bitsandbytes' `NF4_LEVELS`, listed in ascending order) are:

```
[-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
 -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
  0.07958029955625534,  0.16093020141124725,  0.24611230194568634,
  0.33791524171829224,  0.44070982933044434,  0.5626170039176941,
  0.7229568362236023,  1.0]
```

## Task

Implement `qlora_forward`:

```python
def qlora_forward(x: list[list[float]], nf4_codes: list[list[int]], absmax: list[list[float]], blocksize: int, A: list[list[float]], B: list[list[float]], alpha: float) -> list[list[float]]:
    ...
```

* `x` — input activations, shape $(n, d_{\text{in}})$.
* `nf4_codes` — integer array, shape $(d_{\text{out}}, d_{\text{in}})$, each
  entry in $\{0,\dots,15\}$, indexing the NF4 table above.
* `absmax` — float array, shape $(d_{\text{out}}, d_{\text{in}} / \text{blocksize})$,
  the per-row, per-block scale ($d_{\text{in}}$ is guaranteed divisible by
  `blocksize`).
* `blocksize` — number of consecutive input columns sharing one scale.
* `A` — LoRA down-projection, shape $(r, d_{\text{in}})$.
* `B` — LoRA up-projection, shape $(d_{\text{out}}, r)$.
* `alpha` — LoRA scaling numerator; the applied scaling factor is
  $\alpha / r$ with $r$ taken from `A`'s (or `B`'s) shape.

Return $y = x\, W_{\text{eff}}^\top$, shape $(n, d_{\text{out}})$, as defined
by the three steps above.

## Example

```python
# d_out=1, d_in=4, blocksize=2, r=1
nf4_codes = [[7, 15, 0, 8]]       # levels[7]=0.0, levels[15]=1.0, levels[0]=-1.0, levels[8]=0.0796...
absmax    = [[2.0, 3.0]]          # block0 -> cols[0:2], block1 -> cols[2:4]
# dequant row: [0.0*2, 1.0*2, -1.0*3, 0.0796*3] = [0.0, 2.0, -3.0, 0.239...]
A = [[1.0, 0.0, 0.0, 0.0]]
B = [[2.0]]
alpha = 2.0   # scaling = alpha/r = 2.0
# delta row = 2.0 * (B @ A) = [4.0, 0.0, 0.0, 0.0]
# W_eff row  = [4.0, 2.0, -3.0, 0.239...]
x = [[1.0, 1.0, 1.0, 1.0]]
y = qlora_forward(x, nf4_codes, absmax, 2, A, B, alpha)
# y = x @ W_eff.T = sum(W_eff row) ≈ 3.239...
```

## What the gate checks

A single **rel_err** gate builds several random `(x, nf4_codes, absmax,
blocksize, A, B, alpha)` combinations, computes $y$ with a Python oracle that
implements the same three-step formula (dequantize with the NF4 table,
add the scaled LoRA delta, apply the linear layer), and requires your output
to match the oracle's within a relative L2 error of `1e-6`. Any shape
mismatch or exception fails the gate.
