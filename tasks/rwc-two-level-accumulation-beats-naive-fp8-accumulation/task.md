## Context

Long-context inference kernels often keep activations in an 8-bit floating-point
format (FP8) to save memory bandwidth. NumPy has no native FP8 dtype, so here we
use `float16` as a stand-in for "an accumulator register that can only hold
low-precision values" — the numerical failure mode is the same one that shows up
in real FP8 accumulation: as a running sum grows large, the accumulator's fixed
number of mantissa bits can no longer represent the small increments added to it,
and they are silently rounded away ("swamping").

A **naive** accumulator keeps a single low-precision running total for the whole
sequence:

$$
a_0 = 0,\qquad a_i = \operatorname{fp16}\!\big(a_{i-1} + x_i\big)
$$

Once $a_i$ is large, adding a small $x_i$ rounds back to $a_i$ unchanged — every
later term with magnitude below the local FP16 ULP of $a_i$ is lost, and the error
grows with the length of the context.

**Two-level accumulation** fixes this. It never lets a low-precision running total
grow arbitrarily large: it resets a low-precision accumulator at the start of
every fixed-size block, sums only within that (short) block in low precision, then
**promotes** the block's total to `float32` before folding it into a high-precision
running grand total:

$$
\underbrace{b^{(k)}_j = \operatorname{fp16}\!\big(b^{(k)}_{j-1} + x_{kB+j}\big)}_{\text{level 1: fp16 within block }k}
\qquad
\underbrace{T_k = \operatorname{fp32}(T_{k-1}) + \operatorname{fp32}\!\big(b^{(k)}_{B}\big)}_{\text{level 2: fp32 across blocks}}
$$

Because each low-precision run only ever has to represent a sum of at most $B$
terms (instead of $N$ terms), the swamping error stays bounded per block instead
of accumulating over the whole context, and the fp32 cross-block total keeps full
precision where it matters.

## Task

Implement:

```python
def two_level_accumulate(x: np.ndarray, block_size: int) -> float:
    ...
```

* `x` — 1-D NumPy array of `float32` values (the long-context sequence to sum).
* `block_size` — positive `int`, the number of elements per low-precision block.
  The last block may be shorter than `block_size` if `len(x)` is not a multiple
  of it.

Algorithm (must match exactly, so the output matches the oracle bit-for-bit-ish):

1. Split `x` into contiguous blocks of `block_size` elements (last block may be
   shorter).
2. Within each block, accumulate elements **sequentially**, casting the running
   total to `np.float16` after every single addition (`np.float16(prev + x_j)`,
   with `prev` cast up to `float32` before the add so you are always adding two
   `float32` numbers and rounding the result down to `float16`).
3. After a block finishes, cast its final `float16` total to `float32` — this is
   the "dequantize" step — and add it into a running `float32` grand total.
4. Return the grand total as a plain Python `float`.

Do **not** accumulate the whole array in one low-precision running total (that is
the naive strategy this technique is designed to beat), and do not just sum
everything with a single `np.sum(x)` call that skips the block-local low-precision
step entirely — the grader checks the actual numeric result, which depends on
following the two-level procedure above.

## Example

```python
import numpy as np
x = np.array([100.0, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02], dtype=np.float32)

# naive: one running fp16 total over all 8 elements
# -> 100.0   (all seven +0.02 increments get rounded away once the total is 100)

two_level_accumulate(x, block_size=2)
# -> ~100.12   (each pair of 0.02's accumulates in its own short fp16 block,
#               then blocks are folded together in fp32, so far less is lost)

# true sum (float64) is 100.14
```

## What the gate checks

The **rel_err** gate builds several long random sequences (a few thousand
elements, with occasional large-magnitude outliers mixed among many small
values — a realistic pattern for long-context activations) and, for each one,
computes:

* a high-precision reference total (`np.sum(x, dtype=np.float32)`),
* the naive single-accumulator fp16 running sum (for comparison — not gated
  directly),
* your `two_level_accumulate(x, block_size)` result.

For each sequence it computes the error of your result relative to the
reference, scaled by the sum of `|x|` (the natural error scale for a
summation). The gate passes if the **mean** of this scaled error, averaged
across all test sequences, is at or below the threshold — a threshold picked
well below what the naive whole-sequence low-precision accumulator achieves on
the same data, so an implementation that doesn't actually perform the two-level
promote-to-fp32 step will fail it.
