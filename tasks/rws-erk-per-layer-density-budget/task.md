## Context

Given a global sparsity budget for a whole network (say, "60% of all
weights"), how should that budget be split across layers? Splitting it
*uniformly* (every layer gets the same density) is a common baseline,
but it starves small layers — a tiny layer pruned to the same density as
a huge one loses a much larger fraction of its already-scarce
representational capacity. **ERK** (Erdos-Renyi-Kernel), used by RigL
and SET-style dynamic sparse training, instead gives each layer a
density **proportional to its fan-in-plus-fan-out over its parameter
count** — smaller/thinner layers automatically get to keep a higher
density, while large, parameter-heavy layers absorb more of the pruning
— all while the *overall* (parameter-weighted) density still hits the
global target exactly.

### The allocation rule

For layer $l$ with shape $\mathrm{shape}_l$ (a tuple of dimension
sizes), let $n_l = \prod \mathrm{shape}_l$ (parameter count) and
$$
r_l = \left(\frac{\sum \mathrm{shape}_l}{n_l}\right)^{p}
$$
("raw" density weight; $p$ = `erk_power_scale`, default $1$). We want a
single scale factor $\varepsilon$ so that the density
$d_l = \varepsilon \cdot r_l$ satisfies the global parameter-weighted
budget:
$$
\sum_l d_l \, n_l = \text{global\_density} \cdot \sum_l n_l .
$$

The catch: some layers may have such a small $n_l$ (relative to their
$r_l$) that $\varepsilon \cdot r_l$ would exceed $1$ — an impossible
density. Whenever that happens, that layer is pinned to **fully dense**
($d_l = 1$), removed from the pool that $\varepsilon$ is solving for,
and $\varepsilon$ is **recomputed** over the remaining layers using the
budget left over after paying for the now-dense layers. Repeat until no
remaining layer's candidate density exceeds $1$.

## Task

Implement:

```python
def erk_layer_densities(shapes: list[tuple[int, ...]], global_density: float, erk_power_scale: float = 1.0) -> np.ndarray:
    ...
```

* `shapes` — list of per-layer shape tuples (e.g. `(256, 128)` for a
  linear layer).
* `global_density` — target overall (parameter-weighted) density in
  `(0, 1]`.
* `erk_power_scale` — exponent $p$ above (default `1.0`).

Return a 1-D array of per-layer densities $d_l \in (0, 1]$, one per
input shape, following the iterative rule above.

## Example

```python
shapes = [(5, 135), (107, 130), (54, 124)]
d = erk_layer_densities(shapes, global_density=0.223)
# the tiny (5,135) layer's raw ratio is far higher than the other two's,
# so it gets pinned to d=1.0 (fully dense) and epsilon is recomputed
# over the remaining two layers using the leftover budget.
```

## What the gate checks

* **rel_err** — your returned density vector must match a NumPy oracle
  running the exact iterative allocation above to a relative L2 error
  $\le 10^{-6}$, over several random sets of layer shapes and global
  density targets (fixed seed) — including cases where at least one
  layer gets pinned dense.
* **budget_rel_err** — the parameter-weighted average of *your* returned
  densities, $\frac{\sum_l d_l n_l}{\sum_l n_l}$, must match the
  requested `global_density` to within a relative error of $10^{-6}$ (a
  uniform-density solution, or one that ignores the dense-pinning step,
  will violate this on cases with a pinned layer).
