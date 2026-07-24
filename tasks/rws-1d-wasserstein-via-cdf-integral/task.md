## Context

The 1-D Wasserstein-1 (Earth Mover's) distance between two probability
distributions $U$ and $V$ on $\mathbb{R}$ with CDFs $F_U$ and $F_V$ has the
exact closed form

$$
W_1(U,V) = \int_{-\infty}^{\infty} \bigl| F_U(x) - F_V(x) \bigr| \, dx .
$$

For two **empirical** distributions — samples $u \in \mathbb{R}^n$ and
$v \in \mathbb{R}^m$, each point carrying uniform mass $1/n$ or $1/m$ — both
$F_U$ and $F_V$ are step functions, so this integral is exact and finite to
compute (no numerical quadrature needed), even when $n \ne m$:

1. Merge and sort every value from both samples:
   $z_1 \le z_2 \le \dots \le z_{n+m}$ = `sort(concat(u, v))`.
2. Between consecutive merged values $z_k$ and $z_{k+1}$, both $F_U$ and
   $F_V$ are constant, so the integral over that interval is
   $|F_U(z_k) - F_V(z_k)| \cdot (z_{k+1} - z_k)$.
3. $F_U(z_k)$ is the fraction of $u$'s samples that are $\le z_k$ (and
   similarly for $F_V$ and $v$).
4. Summing those rectangle areas over all $n+m-1$ gaps gives $W_1(u,v)$
   exactly:

$$
W_1(u,v) = \sum_{k=1}^{n+m-1} \bigl| F_U(z_k) - F_V(z_k) \bigr| \cdot (z_{k+1}-z_k).
$$

This is the "sorted-diff" / CDF-step algorithm real statistics libraries use
for the unweighted, unequal-sample-size case (rather than, say, padding the
shorter sample with zeros).

## Task

Implement `wasserstein1_cdf_integral`:

```python
def wasserstein1_cdf_integral(u: np.ndarray, v: np.ndarray) -> float:
    ...
```

* `u`, `v` — 1-D `float` arrays, **possibly of different lengths**, each
  treated as an empirical distribution with uniform per-sample weight.

Return $W_1(u,v)$ as a Python `float`, computed exactly via the merged-CDF
step integral above (not by binning/histogramming, and not by padding the
shorter array — the two samples represent independent distributions, not a
paired sequence).

## Example

```python
import numpy as np
u = np.array([0.0, 1.0, 2.0])          # n = 3
v = np.array([0.0, 3.0])               # m = 2 (unequal length)
w1 = wasserstein1_cdf_integral(u, v)
# merged sorted values: 0, 0, 1, 2, 3 (duplicate 0 -> zero-width gap, skip)
# on [0,1): F_u=1/3, F_v=1/2 -> |diff|=1/6, width=1 -> 1/6
# on [1,2): F_u=2/3, F_v=1/2 -> |diff|=1/6, width=1 -> 1/6
# on [2,3): F_u=1.0, F_v=1/2 -> |diff|=1/2, width=1 -> 1/2
# w1 = 1/6 + 1/6 + 1/2 = 0.8333...
```

## What the gate checks

**rel_err** — the grader loads a fixture pair of unequal-length samples
(`w1_u.npy`, a 137-sample bimodal mixture; `w1_v.npy`, a 219-sample shifted
unimodal batch) plus several independently generated random pairs (also of
unequal length, including a couple of edge cases with very small $n$ or
$m$), computes the reference distance with `scipy.stats.wasserstein_distance`
— the real library implementation of this exact CDF-integral algorithm —
and checks the relative error of your result is at most $10^{-6}$ on every
case. Padding the shorter array with zeros (a common but different metric),
binning into a fixed histogram, or forgetting to weight each CDF gap by the
*sample count* rather than treating $u$ and $v$ symmetrically will all
produce a measurable deviation.
