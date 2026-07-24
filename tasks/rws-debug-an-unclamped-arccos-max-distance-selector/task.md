## Context

Given a query vector $q \in \mathbb{R}^d$ and $k$ candidate vectors
$C_0,\dots,C_{k-1} \in \mathbb{R}^d$, a common way to pick the candidate
that points in the *most similar direction* to $q$ is angular distance:

$$
\cos\theta_i = \frac{q \cdot C_i}{\lVert q\rVert\, \lVert C_i\rVert}, \qquad
\theta_i = \arccos(\cos\theta_i),
$$

and the selected candidate is the one with the **smallest** angle,
$\operatorname*{arg\,min}_i \theta_i$.

Two things bite naive implementations of this:

1. **Floating-point overshoot.** For two vectors that are (near-)exact
   positive scalar multiples of each other, $\cos\theta_i$ is
   *mathematically* exactly $1$, but computed in `float64` as
   `dot(q, C_i) / (norm(q) * norm(C_i))` it can come out as
   `1.0000000000000002` — a few ULPs over 1. `np.arccos` on an input
   outside $[-1, 1]$ returns `nan`, silently poisoning the selection.
   Production code always clips the cosine to $[-1, 1]$ before calling
   `arccos`.
2. **Wrong direction of "best".** "Most similar direction" means
   *smallest* angle — `argmin`, not `argmax`. Picking the largest angle
   selects the candidate pointing *away* from the query, the opposite of
   the intended behavior.

## Task

Implement `select_min_angle_block(query, candidates)`:

```python
import numpy as np

def select_min_angle_block(query: np.ndarray, candidates: np.ndarray) -> int:
    ...
```

- `query`: 1-D `float64` array of shape `(d,)`.
- `candidates`: 2-D `float64` array of shape `(k, d)`, `k >= 2`.

Compute the cosine similarity between `query` and every row of
`candidates`, **clip it to `[-1, 1]`**, take `arccos` to get each
candidate's angular distance, and return the `int` index of the candidate
with the **smallest** angular distance to `query`.

## Example

```python
import numpy as np
query = np.array([1.0, 0.0])
candidates = np.array([
    [0.0, 1.0],    # 90 degrees away
    [2.0, 0.0],    # same direction, 0 degrees away
    [-1.0, 0.0],   # 180 degrees away
])
select_min_angle_block(query, candidates)   # 1
```

## What the gate checks

The grader runs several fixed and random cases through an independent
NumPy oracle (clip-then-arccos, then `argmin`) and compares the selected
index. One case is specifically constructed so that the true best
candidate is an exact positive scalar multiple of the query at a scale
where `float64` rounding pushes the raw cosine a few ULPs above `1.0` —
an implementation that skips the clip gets `nan` for that candidate's
distance and mis-selects (or crashes). The remaining cases use
well-separated random vectors that simply catch an `argmax` used where
`argmin` belongs.

The gate metric `argmin_index` is the fraction of cases where your
returned index exactly matches the oracle's; it must equal `1.0` — one
wrong selection on any case fails the gate.
