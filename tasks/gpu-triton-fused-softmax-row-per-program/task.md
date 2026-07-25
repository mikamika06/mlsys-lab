## Context

Triton's block-programming model launches one "program" per unit of
parallel work, and softmax over a batch of rows is the textbook case for
picking that unit to be **one row per program**: every element a program
needs — its whole row — lives in memory it alone touches, no cross-program
communication needed at all.

Softmax itself is shift-invariant: $\mathrm{softmax}(x) =
\mathrm{softmax}(x - c)$ for any constant $c$, because the same $c$ cancels
out of the numerator and denominator of every fraction. That's what makes
the *stable* form safe: subtract the row's own max before exponentiating,
and every exponent is $\le 0$ — never enough to overflow, no matter how
large the raw values are. Skip that subtraction and you're gambling on the
row's actual scale: for this task's fixture, every value is several
hundred, and `exp()` of an un-shifted value that large doesn't just lose
precision — it overflows outright.

## Task

Implement, in `solve.cu`, a kernel with this signature:

```cuda
__global__ void row_softmax(float* out, const float* x, int rows, int cols);
```

Launch with `rows` blocks, 1 thread per block — `blockIdx.x` is the row
this program owns. For its row (`base = blockIdx.x * cols`):

1. `m = ` the max of `x[base .. base+cols-1]`.
2. `sum = ` the sum of `expf(x[base+j] - m)` over the row.
3. `out[base+j] = expf(x[base+j] - m) / sum` for every `j` in `[0, cols)`.

## Example

The grader's fixture draws every value from `[600, 750]` — an ordinary
softmax input, just shifted far from zero. A correctly max-shifted kernel
is completely unaffected by that shift (`max_abs_err` on the order of
`1e-16`, ordinary float64 rounding noise): softmax genuinely doesn't care
what the row's absolute scale is, only its spread.

## What the gate checks

`check.py` builds the large-magnitude fixture, parses `solve.cu`, and runs
`row_softmax` on the software GPU (`arena.cuda_sim.GPU`) with a
`rows`-block, 1-thread-per-block launch. It requires
`max_abs_err <= 1e-6` against a numpy reference. Skipping the max-shift —
summing `expf(x[base+j])` directly — doesn't just produce a slightly wrong
answer on this fixture: `expf` of a raw value around `700` overflows double
precision outright, and the grader reports it as an outright failure
(`max_abs_err = inf`), not a close miss.
