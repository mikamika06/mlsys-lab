## Context

Dynamic sparse training keeps a fixed number of active (nonzero)
connections but periodically reshuffles *which* connections are active:
some active connections are pruned, and an equal number of currently
inactive connections are "regrown" (reactivated). Two classic regrowth
rules differ only in how the new connections are chosen.

Let $M \in \{0,1\}^{n}$ be the current active-connection mask (flattened,
row-major) and $g \in \mathbb{R}^n$ the gradient at every position. Let
$Z = \{i : M_i = 0\}$ be the inactive positions. Both strategies choose a
subset $S \subset Z$ with $|S| = k$ and set $M_i \leftarrow 1$ for every
$i \in S$, leaving all other entries of $M$ unchanged:

- **RigL** picks the $k$ positions in $Z$ with the largest $|g_i|$ —
  it regrows where the gradient signal says a new connection would help
  most, ties broken by the lower index.
- **SET** picks $k$ positions from $Z$ **uniformly at random**, using a
  fixed seed for reproducibility — no gradient information is used at all.

Both leave the active-connection count identical: $|M'| = |M| + k$.

## Task

Implement `regrow_masks(w, g, mask, k, seed)`.

- `w`: dense weight array (not used by the regrowth decision itself; kept
  in the signature because a real DST step also needs it to initialize the
  values of newly grown connections).
- `g`: dense gradient array, same shape as `w`.
- `mask`: boolean array, same shape as `w`. `True` marks a currently active
  connection.
- `k`: number of currently-inactive positions to reactivate.
- `seed`: integer RNG seed used by the SET strategy.

Steps:

1. Flatten `mask` and `g` in row-major (`C`) order. Let `zero_idx` be the
   flat indices where `mask` is `False`, in ascending order.
2. **RigL**: sort `zero_idx` by `|g|` at those positions, descending, with
   ties broken by keeping the lower index first (i.e. a stable sort of
   `-|g|`). Take the first `k` indices and set those positions `True` in a
   copy of the flattened mask.
3. **SET**: draw
   `np.random.default_rng(seed).choice(zero_idx, size=k, replace=False)`
   and set those positions `True` in a fresh copy of the flattened mask.
4. Reshape both results back to `mask`'s original shape.

Return `(rigl_mask, set_mask)`, both boolean arrays with the same shape as
`mask`.

## Example

```python
import numpy as np

mask = np.array([True, False, True, False, False])
g = np.array([0.0, 2.0, 0.0, 5.0, 0.1])
w = np.zeros(5)

rigl_mask, set_mask = regrow_masks(w, g, mask, k=1, seed=0)
# rigl_mask reactivates index 3 (largest |g| among the inactive positions)
# set_mask reactivates one of {1, 3, 4} uniformly at random (seeded)
```

## What the gate checks

The gate rebuilds both strategies with an independent NumPy oracle across
several `(w, g, mask, k, seed)` cases, including one with tied gradient
magnitudes among the inactive positions (to exercise the tie-break rule).

- `exact_match`: for every case, both returned masks must (a) keep every
  originally-active position active, (b) have exactly `k` more active
  positions than the input `mask`, and (c) exactly equal the oracle's
  `rigl_mask` and `set_mask` element-for-element.

A solution that regrows by gradient magnitude for *both* strategies (or
uses a different RNG call/argument order for SET) will diverge from the
oracle's `set_mask` and fail `exact_match`.
