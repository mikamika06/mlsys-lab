## Context

**NF4** (4-bit NormalFloat, used e.g. in QLoRA) quantizes a weight tensor by
first flattening it into a single 1-D stream and splitting that stream into
contiguous, non-overlapping blocks of **64** elements. Every block gets its
own normalization constant — the block's absolute maximum — before its
elements are mapped onto the fixed NF4 code grid:

$$
c_k = \max_{j=0}^{63} \left| w_{64k+j} \right|, \qquad k = 0, 1, \dots, \frac{n}{64}-1,
$$

where $w \in \mathbb{R}^n$ is the flattened (row-major / C-order) weight
tensor and $n$ is a multiple of $64$. This per-block absmax is exactly the
scale later used to bring every element of the block into $[-1, 1]$ before
NF4 code lookup: a block that happens to contain one large weight gets a
correspondingly larger $c_k$, so its other (smaller) elements get less
resolution — the same "shared scale" tradeoff underlying every blockwise
quantization scheme.

## Task

Implement `nf4_block_absmax_scales(W)`:

```python
import numpy as np

def nf4_block_absmax_scales(W: np.ndarray) -> np.ndarray:
    ...
```

`W` is a 2-D `float64` NumPy array (an `(out_features, in_features)` Linear
layer weight matrix) whose total element count is a multiple of `64`.

1. Flatten `W` in row-major (C) order into a length-`n` 1-D stream.
2. Reshape that stream to `(n // 64, 64)`.
3. Return a 1-D `float64` array of length `n // 64` holding each block's
   maximum absolute value.

Use vectorized NumPy — no explicit Python loops.

## Example

```python
import numpy as np
W = np.arange(128, dtype=np.float64).reshape(2, 64) - 32  # 128 elements -> 2 blocks
scales = nf4_block_absmax_scales(W)
# Block 0 is row 0 (values -32..31): max |.| = 32.0
# Block 1 is row 1 (values 32..95):  max |.| = 95.0
# scales == array([32., 95.])
```

## Note on block-vs-row alignment

Blocks are cut from the **fully flattened** stream, not row by row — if
`in_features` is not itself a multiple of 64, a block can straddle the
boundary between two rows. The fixture used for grading has
`in_features` a multiple of 64, so this distinction does not change the
expected output there, but your implementation must flatten first
(`W.reshape(-1)` or `W.ravel()`) rather than compute a per-row max.

## What the gate checks

The grader loads a committed fixture `nf4_w.npy` — a `(96, 256)` matrix
shaped like a real trained Linear layer's weights, with per-output-row
magnitude variation, so the per-block absmax genuinely differs block to
block. The oracle performs the identical flatten -> `reshape(-1, 64)` ->
`np.max(np.abs(.), axis=1)` computation independently in NumPy.

The gate metric is `rel_err`, the global relative L2 error between your
scale vector and the oracle's:

$$
\text{rel\_err} = \frac{\lVert c_{\text{yours}} - c_{\text{oracle}} \rVert_2}{\lVert c_{\text{oracle}} \rVert_2 + 10^{-12}} .
$$

The gate requires `rel_err < 1e-6`. Reshaping in the wrong order (e.g.
Fortran order), taking a per-row instead of per-flattened-block max, or an
off-by-one in the block boundaries will all produce a mismatched scale
vector and fail the gate.
