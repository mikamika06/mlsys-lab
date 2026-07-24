## Context

Self-attention's FLOP cost is quadratic in sequence length. For a single
head of dimension $d$ attending over a sequence of length $n$, computing the
$QK^\top$ score matrix costs $\approx 2dn^2$ FLOPs (a multiply-add per
scalar product term, for every one of the $n^2$ query/key pairs), and
applying those weights to $V$ costs another $\approx 2dn^2$ FLOPs. Summed
over $h$ heads, the total is $C \cdot n^2$ with the fixed per-pair constant

$$
C = 4\,d\,h .
$$

A batch of $B$ variable-length sequences with lengths $n_1,\dots,n_B$ can be
processed two ways:

- **Packed / ragged (varlen) attention** — each sequence only ever computes
  its own $n_i \times n_i$ block. Total FLOPs:
  $$ \text{packed} = C \sum_{i=1}^{B} n_i^2 . $$
- **Padded / dense batching** — every sequence is padded out to
  $L = \max_i n_i$ so the whole batch fits one dense
  $(B, L, L)$ tensor. The kernel computes every pair in that grid, including
  pairs that touch padding tokens. Total FLOPs:
  $$ \text{padded} = C \cdot B \cdot L^2 . $$

Because $\sum n_i^2 \le B \cdot L^2$ (with equality only when every sequence
already has the same length), padding never does *less* work than packing —
and on a batch with a wide length spread, the gap can be enormous. This is
the core reason production inference servers use varlen/ragged attention
kernels instead of naive padded batching.

## Task

Implement `attention_flops`:

```python
def attention_flops(lens: np.ndarray, head_dim: int, num_heads: int) -> tuple[int, int]:
    ...
```

* `lens` — 1-D int array of per-sequence token counts, length $B$.
* `head_dim` — $d$, dimension per attention head.
* `num_heads` — $h$, number of attention heads.

Using $C = 4 \cdot \text{head\_dim} \cdot \text{num\_heads}$, return
`(packed_flops, padded_flops)` as **plain Python `int`s** (these numbers get
large — use exact Python integer arithmetic, not `float`, and not a numpy
integer dtype that could silently overflow):

```
packed_flops = C * sum(n_i ** 2 for n_i in lens)
padded_flops = C * len(lens) * max(lens) ** 2
```

## Example

```python
import numpy as np

lens = np.array([4, 4, 4, 16])   # 3 short sequences + 1 long one
packed, padded = attention_flops(lens, head_dim=64, num_heads=8)
# C = 4*64*8 = 2048
# packed = 2048 * (16 + 16 + 16 + 256) = 2048 * 304
# padded = 2048 * 4 * 16**2 = 2048 * 1024   <- much bigger: 3 sequences
#          are padded all the way out to length 16 for no reason
```

## What the gate checks

A single gate, **exact_match**, compares both returned integers against a
from-scratch reference computed on the same fixture (`lens.npy`, a
deterministic batch of 52 mixed-length sequences) with `head_dim=64`,
`num_heads=8`. Both `packed_flops` and `padded_flops` must match exactly —
no tolerance, since these are exact integer counts, not floating-point
approximations.
