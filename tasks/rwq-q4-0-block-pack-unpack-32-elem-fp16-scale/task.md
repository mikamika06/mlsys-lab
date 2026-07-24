## Context

The ggml **Q4_0** format (used by GGUF / llama.cpp) quantizes a weight
stream in contiguous blocks of **32** elements. Each block stores a single
`float16` scale plus 32 signed 4-bit codes ("nibbles").

For a block $x \in \mathbb{R}^{32}$, let $x^\star$ be the **signed** element
of largest magnitude (i.e. $x^\star = x_{j^\star}$ where
$j^\star = \arg\max_j |x_j|$ — this keeps the sign of whichever element is
most extreme). The scale is

$$
d = \frac{x^\star}{-8},
$$

stored as `float16` (call the round-tripped value $d_{16}$ — every
subsequent computation uses this fp16-precision scale, exactly as real
hardware does, since the scale is read back from its 16-bit storage before
being used to quantize or dequantize). Each element is then mapped onto an
unsigned 4-bit nibble:

$$
q_i = \operatorname{clip}\!\left(\operatorname{round}\!\left(\frac{x_i}{d_{16}}\right) + 8,\; 0,\; 15\right), \qquad i = 0,\dots,31,
$$

and dequantized back with

$$
\hat x_i = (q_i - 8)\, d_{16}.
$$

(For an all-zero block, $d_{16}=0$; by convention every nibble is $8$ and
every dequantized value is $0$.)

## Task

Implement `q4_0_block_pack_unpack(x)`:

```python
import numpy as np

def q4_0_block_pack_unpack(x: np.ndarray) -> dict:
    ...
```

`x` is a 1-D `float64` NumPy array whose length is a multiple of `32`.
Split it into consecutive 32-element blocks and, for every block, compute
the scale and nibbles exactly as defined above. Return a dict:

- `"scale"`: shape `(n_blocks,)`, dtype `float16` — each block's $d_{16}$.
- `"nibbles"`: shape `(n_blocks, 32)`, dtype `uint8`, values in `0..15` —
  each block's 32 quantized codes (no byte-packing — one nibble per array
  entry).
- `"dequant"`: shape `(n_blocks, 32)`, dtype `float64` — `(nibbles - 8) *
  scale` per block, i.e. the round-trip reconstruction of `x`.

## Example

```python
import numpy as np
x = np.zeros(32)
x[5] = -4.0          # the extreme (signed) element of this block
out = q4_0_block_pack_unpack(x)
out["scale"][0]      # np.float16(0.5)   (d = -4.0 / -8 = 0.5)
out["nibbles"][0, 5] # 0   (round(-4.0/0.5)+8 = round(-8)+8 = 0)
out["nibbles"][0, 0] # 8   (round(0/0.5)+8 = 0+8 = 8)
out["dequant"][0, 5] # -4.0   ((0-8)*0.5 = -4.0)
```

## What the gate checks

The grader loads a committed fixture `gguf_w.npy` — a 2560-element stream
(80 blocks of 32) shaped like real quantized-model weights, with block
magnitudes spanning several orders and both positive- and negative-signed
extreme elements across different blocks — and computes the reference
`scale`, `nibbles`, `dequant` with an independent NumPy oracle using the
exact formulas above (`np.float16` cast for the scale, same round+clip
rule for the nibbles).

- `nibble_exact`: the fraction of your `nibbles` entries that exactly
  match the oracle's; gate `== 1.0`. Nibbles are small integers on a fixed
  16-point grid, so there is no meaningful tolerance — a wrong scale sign,
  wrong rounding, or wrong clip range shows up as a mismatched nibble.
- `rel_err`: the global relative L2 error between your `dequant` array and
  the oracle's; gate `< 1e-3` (looser than a typical exact-reconstruction
  gate because the scale is only `float16`-precision — real quantization
  noise at that precision is on the order of $10^{-3}$ relative).

Using the *unrounded* full-precision scale instead of the fp16 round-tripped
one, forgetting the sign of $x^\star$, or an off-by-one in the `+8` shift
will fail `nibble_exact` (and usually `rel_err` too).
