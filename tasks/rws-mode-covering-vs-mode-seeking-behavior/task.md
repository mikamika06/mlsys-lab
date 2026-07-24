## Context

Knowledge distillation losses differ in which direction they compute
KL-divergence, and that choice has a dramatic qualitative effect when the
teacher's distribution is multi-modal and the student can only represent
one mode at a time.

For a teacher $p$ and student $q$ over a shared discrete grid:

$$
D_{\mathrm{fwd}}(p, q) = \mathrm{KL}(p \Vert q) = \sum_x p(x) \log\frac{p(x)}{q(x)}, \qquad
D_{\mathrm{rev}}(p, q) = \mathrm{KL}(q \Vert p) = \sum_x q(x) \log\frac{q(x)}{p(x)},
$$

$$
D_{\mathrm{JSD}}(p, q) = \tfrac{1}{2}\mathrm{KL}(p \Vert m) + \tfrac{1}{2}\mathrm{KL}(q \Vert m), \qquad m = \tfrac{1}{2}(p+q).
$$

Forward-KL is **mode-covering**: because $p(x)\log(p(x)/q(x)) \to \infty$
wherever $q(x)\to 0$ but $p(x) > 0$, a student minimizing forward-KL is
punished severely for placing zero mass anywhere the teacher has mass, so
it "spreads out" to straddle every mode the teacher has, even if that
means sitting in the low-density valley between them.

Reverse-KL is **mode-seeking**: because $q(x)\log(q(x)/p(x)) \to \infty$
wherever $p(x)\to 0$ but $q(x) > 0$, a student minimizing reverse-KL is
punished for placing mass anywhere the teacher doesn't, so it collapses
entirely onto a single mode of the teacher and ignores the rest.

JSD is symmetric in $p, q$, but for well-separated modes it still behaves
like reverse-KL in practice: locking onto one mode is a much better
"local" fit for either KL term than straddling the gap.

## Task

Implement `kd_divergence_family(p, students)`.

- `p`: `float64` array, shape `(G,)`, a discrete probability distribution
  over a fixed grid (a bimodal teacher; already normalized to sum to 1).
- `students`: `float64` array, shape `(n_candidates, G)`, each row a
  candidate single-mode student distribution over the same grid (also
  normalized).

For every candidate student `q = students[i]`, using `eps = 1e-12` inside
every `log` argument to avoid `log(0)`:

- `forward_kl[i] = sum_x p(x) * log((p(x) + eps) / (q(x) + eps))`
- `reverse_kl[i] = sum_x q(x) * log((q(x) + eps) / (p(x) + eps))`
- `m_i = 0.5 * (p + q)`
- `jsd[i] = 0.5 * sum_x p(x) * log((p(x)+eps)/(m_i(x)+eps)) + 0.5 * sum_x q(x) * log((q(x)+eps)/(m_i(x)+eps))`

Return a `dict` with keys `"forward_kl"`, `"reverse_kl"`, `"jsd"`. Each
value is a tuple `(values, argmin)`:

- `values`: `float64` array, shape `(n_candidates,)`, the divergence
  against every candidate student.
- `argmin`: Python `int`, the index of the candidate student that
  minimizes that divergence.

## Example

```python
import numpy as np

x = np.linspace(-8, 8, 161)
def gauss(mu, sigma):
    g = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return g / g.sum()

p = 0.65 * gauss(-3.0, 0.8) + 0.35 * gauss(3.0, 0.8)   # bimodal teacher
p = p / p.sum()
mus = np.linspace(-6, 6, 49)
students = np.stack([gauss(mu, 1.0) for mu in mus])     # single-mode family

out = kd_divergence_family(p, students)
fwd_vals, fwd_argmin = out["forward_kl"]
rev_vals, rev_argmin = out["reverse_kl"]
# mus[fwd_argmin] sits between the two teacher modes (mode-covering)
# mus[rev_argmin] sits almost exactly on the heavier mode (mode-seeking)
```

## What the gate checks

The gate rebuilds two bimodal-teacher / single-mode-student-family
fixtures with an independent NumPy oracle (different mode locations,
mixture weights, and widths).

- `rel_err`: the relative L2 error of your `values` arrays (all three
  divergences, both fixtures) versus the oracle's must be at most `1e-6`.
- `argmin_exact`: every one of your six `argmin` indices (3 divergences x
  2 fixtures) must exactly match the oracle's.
- `pattern_ok`: classifying each `argmin`'s corresponding candidate
  location as "covering" (closer to the midpoint between the two teacher
  modes than to either mode) or "seeking" (closer to one mode), your
  `forward_kl` argmin must classify as covering, and your `reverse_kl` and
  `jsd` argmins must both classify as seeking.

A solution that swaps the arguments of forward-KL and reverse-KL will get
numerically plausible-looking values but land its `forward_kl` argmin on a
single teacher mode instead of between the two, failing `pattern_ok`.
