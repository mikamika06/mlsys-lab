## Context

Activation checkpointing (`torch.utils.checkpoint`) trades compute for
memory: instead of keeping every layer's intermediate activations around for
the backward pass, a checkpointed block discards them after its forward
call and **recomputes** the forward pass when `.backward()` needs them.

That only works if the recomputed forward is bit-for-bit identical to the
original one. A block that contains dropout draws from the process's shared
random-number stream, and by the time `.backward()` runs, many other layers
(or even later training steps) have advanced that same stream. If the
checkpoint does not snapshot the RNG state right before the block's forward
call and restore it right before the recompute, the recomputed dropout mask
$M'$ differs from the original mask $M$, and the backward pass silently
computes the gradient of the *wrong* function.

For a block

$$
h = xW_1,\qquad r=\max(h,0),\qquad d = \frac{r \odot M}{1-p},\qquad y = dW_2 ,
$$

with $M \in \{0,1\}^{n\times k}$ an inverted-dropout mask (keep probability
$1-p$), a correct checkpoint guarantees the mask used to compute $y$ during
the original forward is the exact same mask reused when backpropagating
$\mathrm{d}Y$ into $\mathrm{d}X, \mathrm{d}W_1, \mathrm{d}W_2$.

## Task

Implement `checkpointed_layer` in `solve.py`:

```python
def checkpointed_layer(x, W1, W2, p, seed, n_pre, n_post, dY):
    ...
```

Simulate a single shared random stream `rng = np.random.default_rng(seed)`
threaded through a bigger network:

1. Call `rng.random(3)` exactly `n_pre` times (earlier layers' own
   randomness, run before this block).
2. **Snapshot the RNG state right here** (checkpoint entry), before drawing
   this block's own mask.
3. Draw the dropout mask `mask = (rng.random((x.shape[0], W1.shape[1])) >= p)`
   and run the forward pass above to get `Y`. This is the value that leaves
   the block and continues into the rest of the network; `h`, `r`, `d`,
   `mask` are then discarded (checkpointing does not keep them).
4. Call `rng.random(3)` exactly `n_post` times (later layers, or even
   further training steps, running before `.backward()` is eventually
   called on this block).
5. **Restore the RNG to the state snapshotted in step 2**, then recompute
   the forward pass (regenerating `h`, `r`, `d`, and the identical `mask`)
   and run the analytic backward pass with cotangent `dY`:

$$
\mathrm{d}W_2 = d^{\mathsf T}\mathrm{d}Y,\qquad
\mathrm{d}d = \mathrm{d}Y\,W_2^{\mathsf T},\qquad
\mathrm{d}r = \frac{\mathrm{d}d \odot M}{1-p},\qquad
\mathrm{d}h = \mathrm{d}r \odot \mathbb{1}[h>0],
$$
$$
\mathrm{d}W_1 = x^{\mathsf T}\mathrm{d}h,\qquad
\mathrm{d}X = \mathrm{d}h\,W_1^{\mathsf T}.
$$

Return `(Y, dX, dW1, dW2)`. Use `rng.bit_generator.state` to snapshot and
restore — assigning it back reproduces every subsequent draw from that
generator exactly.

## Example

```python
import numpy as np

n, m, k, o = 4, 3, 5, 2
x = np.random.default_rng(1).standard_normal((n, m))
W1 = np.random.default_rng(1).standard_normal((m, k))
W2 = np.random.default_rng(1).standard_normal((k, o))
dY = np.random.default_rng(1).standard_normal((n, o))

Y, dX, dW1, dW2 = checkpointed_layer(x, W1, W2, 0.4, seed=42, n_pre=2, n_post=3, dY=dY)
# dX, dW1, dW2 must equal the gradient you'd get if the SAME mask drawn
# during the original forward were reused directly for backward -- exactly
# what a correctly-checkpointed recompute reproduces.
```

## What the gate checks

For several `(seed, n_pre, n_post, p)` combinations, the grader computes the
true reference by drawing the mask once (after `n_pre` unrelated draws) and
reusing that exact mask for both the forward output `Y` and the analytic
backward pass — this is the behaviour a correctly save/restored checkpoint
recompute must reproduce. It compares your `(Y, dX, dW1, dW2)` against the
reference with `max_abs_err`, which must be `<= 1e-5`. A checkpoint that
recomputes using whatever RNG state happens to be current (i.e. after the
`n_post` unrelated draws, without restoring) draws a different mask, so its
gradients diverge from the reference by far more than floating-point noise.
