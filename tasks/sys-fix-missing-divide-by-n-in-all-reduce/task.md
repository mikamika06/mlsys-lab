## Context

In synchronous data-parallel training, $N$ workers each compute a local
gradient $g_i$ on their own mini-batch shard. Before the optimizer step,
an **all-reduce** combines them so every worker ends up with the same
gradient — the one that a single worker would have computed on the full
combined batch, on average:

$$
\bar{g} = \frac{1}{N}\sum_{i=1}^{N} g_i.
$$

All-reduce implementations do this as a **sum** across workers (an
efficient ring or tree reduction) followed by a **divide by $N$**. If the
divide step is dropped, every worker silently trains with a gradient
that is $N\times$ too large — equivalent to multiplying the learning
rate by $N$ without meaning to.

## Task

The function below has this exact bug: it sums the gradients but never
divides by the worker count.

```python
def all_reduce_mean_grads(grads: list[np.ndarray]) -> np.ndarray:
    total = np.zeros_like(grads[0], dtype=np.float64)
    for g in grads:
        total += g
    return total  # <-- missing divide by len(grads)
```

Fix `all_reduce_mean_grads` so it returns the correct **mean** gradient
across all `N = len(grads)` workers, matching the shape of each input
gradient.

## Example

```python
grads = [np.array([2.0, 4.0]), np.array([4.0, 8.0])]
all_reduce_mean_grads(grads)
# array([3., 6.])   -- NOT array([6., 12.])
```

## What the gate checks

The grader generates several random sets of `N` gradient tensors
(`np.random.default_rng` seeded, `N` between 2 and 8, random shapes) and
compares your output to `np.mean(np.stack(grads, axis=0), axis=0)`
computed directly with NumPy. `rel_err <= 1e-6` on every case. The
buggy version above returns a vector exactly `N` times too large, so it
fails this gate by construction (its relative error is `N - 1`, since
$\lVert N\bar g - \bar g\rVert / \lVert \bar g\rVert = N-1$).
