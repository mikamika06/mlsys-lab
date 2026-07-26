---
title: "What is softmax vs sigmoid?"
description: "Softmax vs sigmoid explained, with the exact two-class identity measured, where it breaks past two classes, and each function's float32/float64 overflow boundary, plus a graded exercise."
datePublished: 2026-07-26
dateModified: 2026-07-26
author: Oleksandr Savkov
---

# What is softmax vs sigmoid?

Softmax vs sigmoid is not a choice between two different functions but a question of how
many classes are in play: with exactly two, `softmax([a, b])[1]` and `sigmoid(b - a)` are
the same computation, agreeing to `2.220e-16` over 200,000 random pairs — ordinary float64
rounding. Add a third
class and they separate immediately, differing by `0.154942` at `K=3` and by `0.704335` by
`K=100`. Both boundaries, and the exact magnitude at which each function overflows, are
measured below.

## How it works

Sigmoid, `σ(z) = 1 / (1 + e⁻ᶻ)`, is the binary case: one logit, one probability, and its
complement. Softmax, `softmax(x)ᵢ = eˣⁱ / Σⱼ eˣʲ`, is the same idea generalized to `K`
logits competing for one probability distribution. When `K = 2`, the generalization
degenerates back to the special case exactly, by one line of algebra:
`softmax([a, b])[1] = e^b / (e^a + e^b) = 1 / (1 + e^(a-b)) = sigmoid(b - a)`. There is no
approximation in that step — it is the same expression with the shared factor `e^b`
cancelled out of numerator and denominator, which is why the measured difference below is
float64 rounding and nothing else.

What breaks past two classes is not that one line of algebra — it is which question is being
asked. `softmax(x)ᵢ / softmax(x)ⱼ = e^(xᵢ - xⱼ)` for *any* `K`, since every other class's
exponential is a shared factor of `Z` that cancels in the ratio. Dividing that ratio by
itself plus one gives `softmax(x)ᵢ / (softmax(x)ᵢ + softmax(x)ⱼ) = sigmoid(xᵢ - xⱼ)`, also
exactly, for any `K` — but that is the probability of class `i` *conditional on it being `i`
or `j`*, not the raw `softmax(x)ᵢ` this page's table tracks, which shrinks as more classes
compete for the same probability mass. The two questions coincide only when there is nobody
else to compete with.

The numerical failure modes differ for the same structural reason. Sigmoid's one input has
exactly two directions to fail in — `z` very positive or very negative — so a single branch
on the sign of `z` routes each side to the algebraic form whose exponent stays `≤ 0`, and
that is the entire fix; see [`num-branchless-stable-sigmoid`](../tasks/num-branchless-stable-sigmoid/task.md)
for the vectorized version that avoids even that branch. Softmax has no such single sign to
branch on across a `K`-vector, so its fix is the max-shift used throughout
[log-sum-exp](log-sum-exp.md), whose overflow boundary this page reuses — subtracting the
same constant `m = max(x)` from every logit, which is exactly the shift-invariance identity
[proved here](../tasks/llm-shift-invariance-proof-softmax-x-softmax-x-c/task.md). Where the
overflow threshold actually lands depends on the storage format —
[bfloat16 vs float16](bfloat16-vs-float16.md) covers that axis directly — and where a branch
is nearly free on a CPU, the same branch taken differently by the 32 threads of a GPU warp
is [warp divergence](warp-divergence.md), a much more expensive problem wearing the same
`if` statement.

## Softmax equals sigmoid, then stops

Fixing `a = 0`, `b = 1` gives a reference `sigmoid(b - a)`. Growing the class count `K` by
appending `K - 2` extra logits at the same value as `a` and re-measuring `softmax(x)[1]`
against that fixed reference shows exactly where the identity stops holding.

| K (classes) | softmax(x)[1] | \|softmax(x)[1] − sigmoid(b−a)\| |
|---|---|---|
| 2 | 0.731059 | **0.000000** |
| 3 | 0.576117 | 0.154942 |
| 4 | 0.475367 | 0.255692 |
| 5 | 0.404610 | 0.326449 |
| 6 | 0.352187 | 0.378871 |
| 8 | 0.279708 | 0.451351 |
| 16 | 0.153417 | 0.577642 |
| 100 | 0.026724 | 0.704335 |

Reproduce it:

```bash
pip install mlsys-lab
python3 - <<'PY'
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# 1. the two-class identity, checked over many random logit pairs
rng = np.random.default_rng(0)
N = 200_000
a = rng.uniform(-50, 50, N)
b = rng.uniform(-50, 50, N)
sm = np.exp(b) / (np.exp(a) + np.exp(b))
sg = sigmoid(b - a)
print(f"two-class max_abs_diff over {N} pairs: {np.max(np.abs(sm - sg)):.3e}")

# 2. add classes past two and watch the identity break
a0, b0 = 0.0, 1.0
ref = sigmoid(b0 - a0)
print(f"sigmoid(b-a) reference: {ref:.6f}")
for K in (2, 3, 4, 5, 6, 8, 16, 100):
    logits = np.array([a0, b0] + [a0] * (K - 2))
    e = np.exp(logits)
    sm_k = e[1] / e.sum()
    print(f"K={K:>3}  softmax[1]={sm_k:.6f}  abs_diff={abs(sm_k - ref):.6f}")

# 3. exact float32 / float64 boundary where exp() itself overflows to inf
def overflow_boundary(dtype, hi_start):
    lo, hi = np.float64(0.0), np.float64(hi_start)
    with np.errstate(over="ignore"):
        while not np.isinf(dtype(np.exp(hi))):
            hi *= 1.5
        for _ in range(100):
            mid = (lo + hi) / 2
            if np.isinf(dtype(np.exp(mid))):
                hi = mid
            else:
                lo = mid
    return float(hi)

b32 = overflow_boundary(np.float32, 50.0)
b64 = overflow_boundary(np.float64, 500.0)
print(f"float32 exp overflow boundary: {b32}")
print(f"float64 exp overflow boundary: {b64}")

# 4. one fixed formula vs a sign branch: which one actually breaks past that boundary
def unsafe_form(z, dtype):   # exp(z) / (1 + exp(z)) -- exponent grows with z
    z = dtype(z)
    ez = np.exp(z)
    return float(ez / (dtype(1.0) + ez))

def safe_form(z, dtype):     # 1 / (1 + exp(-z)) -- exponent shrinks with z
    z = dtype(z)
    return float(dtype(1.0) / (dtype(1.0) + np.exp(-z)))

with np.errstate(over="ignore", invalid="ignore"):
    z64 = np.float64(b64 + 1.0)
    print("unsafe form value past the boundary:", unsafe_form(z64, np.float64))
    print("safe form value past the boundary:", safe_form(z64, np.float64))
    print("safe form value past minus the boundary:", safe_form(-z64, np.float64))
PY
```

Read the table as two regimes: at `K=2` the difference is `0.000000`, the two forms are the
same function; every class added past that point takes probability mass away from `softmax`
that `sigmoid(b-a)` never learns about, so the gap grows monotonically and has reached
`0.704335` — most of the possible range — by `K=100`. The overflow boundaries tell the other
half: `exp()` itself turns to `inf` at `88.72283908187069` in float32 and
`709.7827128933841` in float64 — the same figure [log-sum-exp](log-sum-exp.md) hits, since
both functions overflow the moment their shared building block, `exp`, does. Past that point
the single formula `exp(z) / (1 + exp(z))` returns `nan` (`inf / inf`), while
`1 / (1 + exp(-z))` returns the exact right answer, `1.0`, and its mirror image at
`-709.7827128933841` returns exactly `0.0` — which is the whole case for branching on the
sign of `z` rather than committing to one algebraic form.

## Practise it

```bash
mlsys grade alg-stable-sigmoid-branch-on-sign
```

[That task](../tasks/alg-stable-sigmoid-branch-on-sign/task.md) gates `max_abs_err <= 1e-12`
against a reference sigmoid over inputs including `-1000` and `1000` directly. The shipped
starter raises `NotImplementedError` and fails outright. The one-line fix that looks
equally natural to the branching one — `exp(z) / (1 + exp(z))` for every `z`, with no
sign check — passes for `z = -1000` but fails at `z = 1000`: `exp(1000)` overflows past the
`709.7827128933841` boundary measured above, giving `inf / (1 + inf) = nan`, and the gate
sees an infinite error where the branching version sees exactly `1.0`.

More tasks in the same area, roughly increasing in scope:
[prove softmax's shift invariance](../tasks/llm-shift-invariance-proof-softmax-x-softmax-x-c/task.md)
(the algebra behind the max-shift, no numerics),
[stable softmax via max-subtraction](../tasks/alg-naive-softmax-overflow-max-subtraction-fix/task.md)
(the softmax-side counterpart to the sigmoid task above),
[branchless stable sigmoid](../tasks/num-branchless-stable-sigmoid/task.md) (vectorize away
even the sign check),
[log-sigmoid and its gradient](../tasks/num-logsigmoid-gradient/task.md) (the log-space form
used inside binary cross-entropy), and
[softmax survives 300 large logits](../tasks/llm-stable-softmax-survives-300-logits/task.md)
(the K-class stability gate, scaled up).

## Common mistakes

- **Using `sigmoid(b - a)` as a stand-in for `softmax(x)[b]` once there is a third class.**
  The measured table above shows the gap is not small: by `K=8` it is already `0.451351`,
  larger than the value itself. The two-class identity does not partially generalize — it
  stops applying the moment a third logit exists.
- **Picking one sigmoid formula and shipping it.** `1 / (1 + exp(-z))` alone silently warns
  on very negative `z` but still returns the right answer; `exp(z) / (1 + exp(z))` alone
  returns outright `nan` past `709.7827128933841` on positive `z`. They look
  interchangeable and are not — only the branch keeps the exponent's argument `≤ 0` on both
  sides.
- **Reaching for a max-shift inside a plain sigmoid.** The shift subtracts the same constant
  from every element of a vector of competing logits; a lone `z` has nothing to subtract
  from except itself, which changes nothing. The sign branch is sigmoid's actual analogue
  of softmax's shift, not a smaller version of the same trick.

## Where else to practise this

From the [full survey of what exists](../LANDSCAPE.md) for this area:

- **[CS231n Assignment 1](https://cs231n.github.io/assignments2026/assignment1/)** — a
  learner implements a softmax classifier and its loss by hand, gradient-checked in the
  notebook. Covers the multi-class side; it does not touch the K=2 identity or either
  overflow boundary.
- **[Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction)**
  — the logistic-regression labs build sigmoid from scratch, the closest free,
  actively-maintained treatment of the binary side; graded for a certificate, not against an
  adversarial overflow fixture.
- **[Float Exposed](https://float.exposed/)** — flip the bits of a float64 by hand and watch
  it become `inf`; the best way to *see* why `709.7827128933841` is where `exp` gives out,
  with no softmax or sigmoid content of its own.
- **[fp-conv](https://sw23.github.io/fp-conv/)** — the same bit-level exploration extended to
  bf16 and the fp8 formats, useful once a mixed-precision softmax or sigmoid runs in a
  narrower format and the overflow boundary measured here moves accordingly.
- The survey's own conclusion here is blunt: nowhere found "grades a stable-softmax/
  log-sum-exp implementation against an overflow-triggering input" the way the tasks above
  do — the softmax-sigmoid identity itself appears ungraded anywhere surveyed.

## References

1. Bishop, C., *Pattern Recognition and Machine Learning*, §4.2 — softmax as the
   multi-class generalization of the logistic sigmoid.
   https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/
2. Goldberg, D., *What Every Computer Scientist Should Know About Floating-Point
   Arithmetic*, 1991 — the overflow mechanics behind both boundaries measured here.
   https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html
3. NumPy documentation, `numpy.exp` — IEEE 754 overflow behavior (`inf` on overflow, a
   `RuntimeWarning`, not an exception). https://numpy.org/doc/stable/reference/generated/numpy.exp.html
