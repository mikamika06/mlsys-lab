## Context

**N:M structured sparsity** (as used by NVIDIA's sparse tensor cores) keeps
exactly 2 of every 4 consecutive weights and zeros the other 2. For one
group of 4 weights there are exactly $\binom{4}{2} = 6$ possible "keep"
patterns — which 2 of the 4 positions survive. In a fixed canonical order
(lexicographic by the pair of kept indices), the 6 patterns are:

| $k$ | kept indices | mask |
|---|---|---|
| 0 | $\{0,1\}$ | $[1,1,0,0]$ |
| 1 | $\{0,2\}$ | $[1,0,1,0]$ |
| 2 | $\{0,3\}$ | $[1,0,0,1]$ |
| 3 | $\{1,2\}$ | $[0,1,1,0]$ |
| 4 | $\{1,3\}$ | $[0,1,0,1]$ |
| 5 | $\{2,3\}$ | $[0,0,1,1]$ |

A probabilistic (not-yet-hardened) mask-selection scheme scores each group
with a probability distribution $p \in \mathbb{R}^6$ over these 6 patterns
($\sum_k p_k = 1$) instead of committing to one. Two useful statistics
summarize a group's distribution before it is hardened into a discrete
choice:

$$
\mathbb{E}[\text{density}] = \sum_{k=0}^{5} p_k \cdot \frac{\lVert \text{mask}_k \rVert_1}{4},
\qquad
\mathbb{E}\!\left[\textstyle\sum |w|_{\text{retained}}\right] = \sum_{k=0}^{5} p_k \sum_{i=0}^{3} \text{mask}_k[i]\, |w_i| .
$$

Because every one of the 6 patterns keeps exactly 2 of 4 elements, the
first quantity is a structural invariant: $\mathbb{E}[\text{density}] = 0.5$
for *any* valid probability distribution over the 6 patterns — a useful
sanity check on the aggregation itself. The second quantity is **not**
constant: it depends on how the probability mass lines up with which
positions hold the largest weight magnitudes, and is exactly what such a
scheme would use (e.g. as a regularizer) to encourage probability mass onto
patterns that retain more total magnitude.

## Task

Implement `expected_pattern_stats`:

```python
def expected_pattern_stats(p: list[list[float]], w: list[list[float]]) -> tuple[list[float], list[float]]:
    ...
```

* `p` — `float` array of shape $(G,6)$: for each of $G$ groups, a
  probability distribution over the 6 canonical patterns above (each row
  sums to 1, in the table's $k$-order).
* `w` — `float` array of shape $(G,4)$: for each group, the **absolute**
  magnitude $|w_i|$ of its 4 weights.

Return `(expected_density, expected_retained)`, each a 1-D `float64` array
of length $G$:
* `expected_density[g]` — $\mathbb{E}[\text{density}]$ for group $g$
  (always $0.5$ for a valid `p`, but compute it via the formula above, not
  a hardcoded constant).
* `expected_retained[g]` — $\mathbb{E}[\sum|w|_{\text{retained}}]$ for
  group $g$, from the formula above.

## Example

`p = [0.5, 0.5, 0, 0, 0, 0]` (half the mass on pattern $\{0,1\}$, half on
$\{0,2\}$), `w = [1.0, 2.0, 3.0, 4.0]`:
pattern $\{0,1\}$ retains $1.0+2.0=3.0$; pattern $\{0,2\}$ retains
$1.0+3.0=4.0$. `expected_density = 0.5`,
`expected_retained = 0.5\cdot 3.0 + 0.5\cdot 4.0 = 3.5`.

## What the gate checks

**rel_err** — the grader loads a fixture batch (`pat_p.npy`, `pat_w.npy`:
50 groups with `p` drawn from varied Dirichlet distributions and `w` drawn
from scaled Gaussian magnitudes) plus a couple of independently generated
synthetic batches (including a hand-checkable single-group case), computes
`(expected_density, expected_retained)` independently with a Python oracle,
and checks the global relative L2 error between your concatenated
`(expected_density, expected_retained)` output and the oracle's is at most
$10^{-6}$. Because `expected_density` alone is always $0.5$, a solution
that hardcodes it without correctly aggregating `expected_retained` from
`p` and `w` will fail this combined check even though half of it "looks"
right.
