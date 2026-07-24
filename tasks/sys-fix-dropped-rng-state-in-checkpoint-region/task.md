## Context

Activation checkpointing trades memory for compute: instead of keeping a
block's internal activations around for the backward pass, the framework
discards them after the forward pass and **recomputes** the block later,
right before it's needed.

This is exact only if the recomputed forward is *bit-identical* to the
original one. A block containing dropout draws from a random stream, so
identical recomputation requires the checkpoint to snapshot the shared
RNG's state right before the block consumes any randomness, and restore
that snapshot before recomputing — otherwise the recompute inherits
whatever state the RNG has drifted to after every other layer that shared
the same stream has also drawn from it in between.

For a block `linear -> relu -> dropout(p)`:

$$
h = \max(xW,\, 0), \qquad
y = h \odot \frac{m}{1-p}, \qquad
m_{ij} \sim \mathrm{Bernoulli}(1-p)
$$

where $m$ is drawn from the shared RNG stream. If the checkpoint restores
the RNG correctly, the mask $m$ used for the recompute is identical to the
one used in the original forward, so $y_{\text{recomputed}} = y_{\text{forward}}$
exactly. If it doesn't, $m$ differs and the two outputs diverge.

## Task

Fix the buggy `checkpointed_dropout_block`:

```python
def checkpointed_dropout_block(x, W, p, seed, n_pre, n_between):
    ...
```

- `x`: `(n, d)` float64 input.
- `W`: `(d, k)` float64 weight matrix.
- `p`: dropout probability, `0 <= p < 1`.
- `seed`: seeds a fresh `np.random.default_rng(seed)` shared by the whole
  (simulated) network.
- `n_pre`: number of unrelated `rng.random(3)` draws made by earlier
  layers *before* this block runs, on the same shared RNG.
- `n_between`: number of unrelated `rng.random(3)` draws made by later
  layers *between* this block's forward call and its later recompute (the
  rest of the network's forward pass running before backward reaches this
  block), also on the same shared RNG.

Return `(y_forward, y_recomputed)`:

- `y_forward`: the block's actual forward output.
- `y_recomputed`: the output produced when the block is recomputed later
  (as an activation-checkpointed block does during backward), using a mask
  drawn from the *same shared RNG object*, after `n_between` unrelated
  draws have already happened on it.

The bug: the current code snapshots nothing. It draws the forward mask,
lets `n_between` unrelated draws happen, then draws the recompute mask
straight from wherever the stream now sits — a different mask than the
one the forward pass used, so `y_recomputed != y_forward`.

Fix it by snapshotting `rng.bit_generator.state` right before the
forward's own dropout draw, and restoring that snapshot right before the
recompute's dropout draw, so the recompute mask is provably identical to
the forward mask regardless of `n_between`.

## Example

```python
import numpy as np

x = np.random.default_rng(0).standard_normal((3, 2))
W = np.random.default_rng(1).standard_normal((2, 4))

y_fwd, y_recomp = checkpointed_dropout_block(x, W, p=0.5, seed=7, n_pre=1, n_between=10)
# A correct implementation gives y_fwd == y_recomp exactly, no matter how
# large n_between is, because the RNG is restored before the recompute's
# own dropout draw.
```

## What the gate checks

The grader builds several seeded configurations (varying `p`, `n_pre`,
`n_between`) and computes both the true forward output and the true
recomputed output directly from the shared-RNG mechanics described above
(snapshot before the block's draw, restore before the recompute's draw).

`max_abs_err` is the worst-case max elementwise absolute difference
between your `(y_forward, y_recomputed)` and this reference pair, across
all configurations (must be `< 1e-6`). A checkpoint that drops the RNG
snapshot produces a `y_recomputed` that drifts further from `y_forward`
as `n_between` grows, which this gate catches directly — it does not
merely check that your two outputs agree with each other, it checks both
against an independently computed oracle mask sequence.
