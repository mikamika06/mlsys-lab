## Context

Optimal transport (Wasserstein distance) between two mass distributions
$p, q$ on the same $n$ support points $x_1,\dots,x_n\in\mathbb{R}$ with
ground cost $c(x,y) = |x-y|$ generally requires solving a linear
program (find the min-cost coupling moving $p$'s mass onto $q$'s
mass). But in **one dimension**, with a convex ground cost like
$|x-y|$, the optimal coupling is always *monotone* (never crosses
itself), which collapses the LP to a closed form: sort the support
points, take cumulative masses, and integrate their absolute
difference:

$$
\text{OT}(p,q) = \sum_{k=1}^{n-1} \big|P_k - Q_k\big| \cdot (x_{(k+1)} - x_{(k)})
$$

where $x_{(1)}<\dots<x_{(n)}$ are the support points sorted ascending,
and $P_k = \sum_{j\le k} p_{(j)}$, $Q_k=\sum_{j\le k} q_{(j)}$ are
cumulative masses in that same sorted order ($p_{(j)}$, $q_{(j)}$
being $p,q$ reordered to match the sort). This is the discrete
analogue of $\int |F_p(t)-F_q(t)|\,dt$, the CDF-difference formula for
the 1-Wasserstein distance — no LP solver needed.

## Task

Implement `ot_cost_1d`:

```python
def ot_cost_1d(positions: list[float], p: list[float], q: list[float]) -> float:
    ...
```

- `positions`: 1-D `float64` array of `n` distinct support point locations (not necessarily sorted).
- `p`, `q`: 1-D nonnegative `float64` arrays of length `n`, masses at those positions. `sum(p) == sum(q)` (equal total mass).

1. Sort the support points ascending, and reorder `p` and `q` to match
   that same order.
2. Compute cumulative sums `P`, `Q` of the reordered `p`, `q`.
3. Return `sum_{k=0}^{n-2} |P[k] - Q[k]| * (x_sorted[k+1] - x_sorted[k])`.

## Example

```python
positions = [3.0, 1.0, 2.0]   # unsorted
p = [0.0, 1.0, 0.0]           # all mass at position 1
q = [0.0, 0.0, 1.0]           # all mass at position 3
cost = ot_cost_1d(positions, p, q)
# sorted positions: [1, 2, 3]; moving all mass from 1 to 3 costs |1-3| = 2.0
```

## What the gate checks

The grader builds several small (`n <= 8`) seeded `(positions, p, q)`
cases and computes the reference cost by solving the transportation
linear program *exactly* with `scipy.optimize.linprog` (minimize
`sum_ij |x_i - x_j| * flow_ij` subject to the row/column mass
constraints) — a genuinely independent ground truth, not the
cumulative-sum formula itself.

`rel_err` is the worst-case relative error between your `ot_cost_1d`
output and the LP-optimal cost, across all cases (must be `<= 1e-6`)
— since 1-D optimal transport with a convex ground cost provably
equals the closed-form CDF-difference formula, a correct
implementation matches the LP to numerical precision; a wrong sort
order, an unsorted cumulative sum, or a missing gap-weighting term
produces a cost visibly higher than the true LP optimum.
