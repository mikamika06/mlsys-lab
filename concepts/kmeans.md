---
title: "What is kmeans?"
description: "Kmeans explained, with a measured table of iterations-to-convergence and final inertia for random vs k-means++ init that you can reproduce, plus graded exercises."
datePublished: 2026-07-26
dateModified: 2026-07-26
author: Oleksandr Savkov
---

# What is kmeans?

Kmeans is an iterative clustering algorithm that partitions $n$ points into $k$ groups by
alternately assigning each point to its nearest centroid and moving each centroid to the
mean of the points assigned to it. Because that loop is a local search, not a global one, two
runs on the identical dataset can converge to inertia values that differ by 67% depending only
on which points happened to seed the centroids. Below, ten seeded runs of both random and
k-means++ initialisation, and how much the seeding actually buys you.

## How it works

Lloyd's algorithm, the thing everyone means by "kmeans", is two steps repeated until nothing
changes. **Assignment**: for every point, compute its squared distance to all $k$ centroids
and take the argmin — this is
[an assignment step you can implement in isolation](../tasks/alg-k-means-assignment-step-only/task.md).
**Update**: replace each centroid with the mean of the points currently assigned to it —
[recoverable from labels and data alone](../tasks/alg-recover-centroids-from-assignments-data/task.md).
Each full pass provably never increases the total squared distance from points to their
centroid, called the inertia; that non-increase is provable directly, not just observed
empirically, which is
[a proof exercise in its own right](../tasks/alg-prove-inertia-non-increasing/task.md). The
loop stops when assignments stop changing, or at a fixed iteration cap — predicting exactly
which iteration that will be, and what the final labels look like, is
[a separate skill from implementing the loop](../tasks/alg-predict-convergence-iteration-final-labels/task.md).

None of this says where the first $k$ centroids come from, and that choice is the whole
subject of this page. **Random init** picks $k$ existing points uniformly at random. **k-means++**
picks the first point uniformly, then each subsequent point with probability proportional to
its squared distance from the nearest centroid already chosen, which is a weighted sample you
can drive from an explicit random stream instead of a black-box RNG —
[exactly what one task asks you to do](../tasks/alg-k-means-seeding-from-given-random-stream/task.md).
Because kmeans only ever finds a local optimum, initialisation is not a footnote: a bad draw
can strand a centroid on top of another one, or leave an entire cluster with none, which forces
a repair rule for [the empty-cluster edge case](../tasks/alg-debug-empty-cluster-crash-nan-centroid/task.md).

Two costs are worth naming because they show up elsewhere in this bank. The naive assignment
step is $O(nkd)$ with an explicit double loop; the same expansion identity that makes it a
handful of matrix operations —
[an exercise on its own](../tasks/alg-vectorize-assignment-with-expansion-trick/task.md) — is
the same broadcast-and-reduce shape whose memory-access pattern
[memory coalescing](memory-coalescing.md) counts on a GPU. And a parallel implementation that
accumulates each cluster's running sum in a small shared array is exactly the layout
[false sharing](false-sharing.md) punishes: eight cluster accumulators packed into one cache
line means eight threads fighting over ownership of it, even though the clusters are disjoint.
At large $n$, [mini-batch kmeans](../tasks/alg-mini-batch-k-means-deterministic/task.md) trades
the full $O(nkd)$ pass for a fixed-size random subset per step — a different lever than
initialisation, and one that trades exactness for speed rather than local-optimum risk.

## Iterations and inertia measured against the seed

The same 320-point, 8-blob dataset, fixed once, run from ten different seeds. Each seed drives
both a random-init run and a k-means++ run of the identical Lloyd loop; the only thing that
differs between the two columns is which points started as centroids.

| seed | random: iters / inertia | k-means++: iters / inertia |
|---|---|---|
| 0 | 17 / 423.7 | 15 / 428.6 |
| 1 | 12 / 485.6 | 9 / 423.1 |
| 2 | 10 / 417.1 | 8 / 417.2 |
| 3 | 8 / **688.0** | 10 / 442.8 |
| 4 | 21 / 483.5 | 12 / 413.2 |
| 5 | 12 / 412.9 | 12 / 437.3 |
| 6 | 13 / 493.3 | 8 / 437.0 |
| 7 | 9 / 483.5 | 9 / 417.5 |
| 8 | 12 / 552.9 | 12 / 442.4 |
| 9 | 12 / 413.2 | 11 / 442.2 |
| **mean ± std** | 12.6 ± 3.6 / 485.4 ± 80.4 | **10.6 ± 2.1** / **430.1 ± 11.1** |

Reproduce it:

```bash
pip install mlsys-lab
python3 - <<'PY'
import numpy as np

def make_blobs(seed, k, per=40, d=2, spread=1.0, sep=5.0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-sep, sep, size=(k, d))
    return np.concatenate([centers[j] + spread * rng.standard_normal((per, d)) for j in range(k)])

def kpp_init(X, k, rng):
    idx = [rng.integers(len(X))]
    centers = X[idx]
    for _ in range(1, k):
        d2 = np.min(((X[:, None] - centers[None]) ** 2).sum(-1), axis=1)
        idx.append(rng.choice(len(X), p=d2 / d2.sum()))
        centers = X[idx]
    return centers.copy()

def lloyd(X, centers, max_iter=300):
    prev = None
    for it in range(1, max_iter + 1):
        labels = np.argmin(((X[:, None] - centers[None]) ** 2).sum(-1), axis=1)
        centers = np.array([X[labels == j].mean(0) if np.any(labels == j) else centers[j]
                             for j in range(len(centers))])
        if prev is not None and np.array_equal(labels, prev):
            return centers, labels, it
        prev = labels
    return centers, labels, max_iter

def inertia(X, c, lab):
    return float(np.sum((X - c[lab]) ** 2))

K = 8
X = make_blobs(seed=0, k=K)                       # 320 points, 8 blobs, fixed once
r_it, r_in, k_it, k_in = [], [], [], []
for s in range(10):
    rand_centers = X[np.random.default_rng(s).choice(len(X), K, replace=False)]
    c, lab, it = lloyd(X, rand_centers)
    r_it.append(it); r_in.append(inertia(X, c, lab))
    c, lab, it = lloyd(X, kpp_init(X, K, np.random.default_rng(s)))
    k_it.append(it); k_in.append(inertia(X, c, lab))
    print(f"seed={s}  random={r_it[-1]:>2} iters / {r_in[-1]:>6.1f} inertia   "
          f"kmeans++={k_it[-1]:>2} iters / {k_in[-1]:>6.1f} inertia")

r_it, r_in, k_it, k_in = map(np.array, (r_it, r_in, k_it, k_in))
print(f"random   inertia mean={r_in.mean():.1f} std={r_in.std():.1f}")
print(f"kmeans++ inertia mean={k_in.mean():.1f} std={k_in.std():.1f}")
PY
```

`mlsys grade` on the task below checks a `compare_inertia` implementation the same shape as
this script, gated on `rel_err`, so the exact numbers you get back are the exact numbers a
grader is comparing you against. Read the table by its spread, not just its mean:
**k-means++ wins outright on inertia in 6 of 10 seeds and on iteration count in 6 of 10** — a
real edge, but not the landslide the folklore suggests. What it actually buys is a much
tighter worst case: random init's inertia ranges from 412.9 to 688.0 (a 67% spread), while
k-means++'s ranges from 413.2 to only 442.8. Seed 3 is the whole story — random init drew two
starting centroids inside the same blob, stranded a whole other blob without one, and Lloyd's
algorithm converged fast (8 iterations) to a bad local optimum instead of slowly to a good one.
k-means++'s distance-weighted sampling makes that specific failure much less likely, not
impossible.

## Practise it

```bash
mlsys grade alg-random-init-vs-k-means-final-inertia
```

[That task](../tasks/alg-random-init-vs-k-means-final-inertia/task.md) gates `rel_err <= 1e-06`
against a reference `compare_inertia(X, n_clusters, seeds)` that returns both initialisation
strategies' final inertia per seed. The shipped starter is `raise NotImplementedError`, so it
fails immediately; the interesting way to fail past that is to get Lloyd's loop itself right
but seed k-means++ with the wrong random stream — `np.random.default_rng(seed).choice` at the
wrong point in the sequence produces a different, still-valid clustering, and `rel_err` catches
the mismatch even though your output looks reasonable in isolation.

In roughly increasing order:
[the assignment step alone](../tasks/alg-k-means-assignment-step-only/task.md),
[full Lloyd from a fixed initialisation](../tasks/alg-full-lloyd-with-fixed-init/task.md),
[k-means++ seeding from a given random stream](../tasks/alg-k-means-seeding-from-given-random-stream/task.md),
[repairing an empty cluster deterministically](../tasks/alg-debug-empty-cluster-crash-nan-centroid/task.md),
and [mini-batch kmeans](../tasks/alg-mini-batch-k-means-deterministic/task.md) for the
large-$n$ variant.

## Common mistakes

- **Treating "kmeans converged" as "kmeans found the best clustering".** Convergence means
  the assignment step stopped changing; inertia is provably non-increasing along the way, but
  nothing prevents it from converging to a local optimum 67% worse than another run's, as
  seed 3 above shows directly.
- **Re-seeding the RNG mid-run.** k-means++'s guarantee depends on drawing every centroid from
  one continuous weighted stream. Reseeding between draws, or reusing the same seed for both
  the initial point and the weighted picks, silently degrades it toward random init while
  looking identical in the source.
- **Forgetting the empty-cluster case.** A cluster with zero points makes its mean a division
  by zero. It is rare on a 320-point toy dataset and routine on real, unevenly dense data —
  code that has never hit it has not been tested, not been proven correct.
- **Assuming k-means++ removes the need for multiple restarts.** It shrinks the bad-outcome
  tail (max inertia 442.8 instead of 688.0 here) but does not eliminate it — seed 5 above still
  landed k-means++ at a *higher* inertia than random init got by luck. Production kmeans (e.g.
  scikit-learn's default) still runs several inits and keeps the best.

## Where else to practise this

From the [full survey of what exists](../LANDSCAPE.md) for this track:

- **[deep-ml.com](https://www.deep-ml.com/problems)** — the closest analog in spirit: a
  browser judge with a k-means problem among 100+, graded pass/fail against hidden tests. Real
  grading, but a single problem rather than the seeding/edge-case/vectorisation spread this
  page links to.
- **[Machine Learning Specialization (Andrew Ng)](https://www.coursera.org/specializations/machine-learning-introduction)**
  — has a named, auto-graded k-means lab as part of a paid ($49/mo) certificate track; guided
  fill-in-the-blank rather than implement-from-a-spec, and it does not touch k-means++ versus
  random init at all.
- **[Data Science from Scratch (Joel Grus)](https://github.com/joelgrus/data-science-from-scratch)**
  — builds kmeans with plain Python lists, no NumPy, specifically to force understanding of
  every step. No grader; it is a book to read alongside this page, not a substitute for it.
- **[rushter/MLAlgorithms](https://github.com/rushter/MLAlgorithms)** — clean, minimal NumPy
  kmeans among a dozen classic algorithms, still receiving commits as of 2026-05. Read-only
  reference code, useful for seeing a production-lean style once you have your own version
  passing the gate.
- **[trekhleb/homemade-machine-learning](https://github.com/trekhleb/homemade-machine-learning)**
  — pairs a from-scratch kmeans with an interactive notebook walking the underlying math. No
  tests or grading, but the best of this group for building intuition before you write code
  that has to pass one.

## References

1. Arthur, D. and Vassilvitskii, S., *k-means++: The Advantages of Careful Seeding*, SODA 2007.
   https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf
2. Lloyd, S., *Least Squares Quantization in PCM*, IEEE Transactions on Information Theory,
   1982 (circulated as a Bell Labs technical note in 1957).
   https://www.cs.toronto.edu/~roweis/csc2515-2006/readings/lloyd57.pdf
3. scikit-learn documentation, *k-means: `init` parameter and `n_init`*, on why the default
   still runs several restarts even with k-means++ seeding.
   https://scikit-learn.org/stable/modules/clustering.html#k-means
