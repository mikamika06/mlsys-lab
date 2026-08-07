## Context

Structured width-pruning tools (e.g. LLM-Pruner-style hidden-size search)
shrink a transformer's hidden width $d$ to hit a target parameter budget.
Cutting $d$ doesn't touch just one tensor: every tensor whose shape depends
on the hidden width — QKV projections, the attention output projection,
the MLP's up/gate and down projections, every layernorm weight, the token
embedding, the LM head — must be sliced **together** ("coupled slicing"),
each to its own shape at that width. The actual parameter count saved is
only correct if *every* coupled tensor is accounted for, not just the
obviously-square ones.

Each coupled tensor $t$ is described generically: at hidden width $d$, its
2-D shape is

$$
\text{shape}_t(d) = \big(c^{(0)}_t d + k^{(0)}_t,\ \ c^{(1)}_t d + k^{(1)}_t\big),
$$

where $c^{(0)}_t, k^{(0)}_t, c^{(1)}_t, k^{(1)}_t$ are fixed integers (a
1-D tensor of length $d$, such as a layernorm weight, is represented with
its second axis fixed at $c^{(1)}=0,\ k^{(1)}=1$). The total parameter
count of the whole (coupled) model at width $d$ is

$$
P(d) = \sum_{t} \text{shape}_t(d)_0 \cdot \text{shape}_t(d)_1 .
$$

Given a discrete set of candidate widths $\mathcal{D}$ (a search grid) and
a parameter budget $B$, the width-search picks

$$
d^\star = \max\{\, d \in \mathcal{D} : P(d) \le B \,\}
$$

— the **largest** candidate width whose exact coupled parameter count still
fits the budget. (If no candidate fits, fall back to the candidate with the
smallest $P(d)$.)

## Task

Implement `pick_width_for_budget`:

```python
def pick_width_for_budget(coefs: list[list[int]], consts: list[list[int]], widths: list[int], budget: int) -> tuple[int, int]:
    ...
```

* `coefs`, `consts` — integer arrays of shape $(T,2)$: for tensor $t$,
  `coefs[t] = (c0, c1)` and `consts[t] = (k0, k1)`, so its shape at width
  $d$ is `(c0*d + k0, c1*d + k1)`.
* `widths` — 1-D integer array of candidate widths $\mathcal{D}$, in any
  order.
* `budget` — integer, the maximum allowed total parameter count.

For every candidate width, compute $P(d)$ (summed over **all** $T$
tensors). Return `(chosen_width, param_count)`:
* `chosen_width` — the largest $d \in \mathcal{D}$ with $P(d) \le B$ (or,
  if none satisfies that, the $d$ with the smallest $P(d)$).
* `param_count` — the exact $P(\text{chosen\_width})$, as a Python `int`.

## Example

Two tensors: a $(3d, d)$ projection and a $(d,)$ bias, so
$P(d) = 3d^2 + d$. Candidates $\mathcal{D} = \{2, 3, 4\}$, budget $B=40$:
$P(2)=14$, $P(3)=30$, $P(4)=52$. The largest width with $P(d)\le 40$ is
$d=3$, so the answer is `(3, 30)`.

## What the gate checks

**exact_match** — the grader loads a fixture describing a 12-layer
LLaMA-style stack (combined QKV, attention output proj, SwiGLU gate+up,
MLP down proj, two layernorms per layer, plus embedding, untied LM head,
and a final norm — 75 coupled tensors total) with a 61-point width search
grid and a fixed budget, plus a couple of independently constructed
synthetic tensor sets, computes the oracle's `(chosen_width, param_count)`
for each by sweeping every candidate and summing every tensor's exact
shape, and checks your `(chosen_width, param_count)` matches **exactly**
(integer equality on both). Forgetting even one coupled tensor (e.g. a
layernorm, or only counting one of the two MLP projections), rounding a
non-square shape, or picking the largest width regardless of the budget
will all produce a different pair.
