---
title: "What is log sum exp?"
description: "Log sum exp explained, with the exact float64 overflow and underflow boundaries measured against the max-shift trick, plus a graded exercise."
datePublished: 2026-07-26
dateModified: 2026-07-26
author: Oleksandr Savkov
---

# What is log sum exp?

Log sum exp — one word, `logsumexp`, in every library that ships it
(`scipy.special.logsumexp`, `torch.logsumexp`) — is the function
log(Σᵢ eˣⁱ), the operation that turns a vector of log-scale scores back into
a single log-scale total: the normalizer behind softmax, cross-entropy, and
mixture-model likelihoods. Computed literally, it
overflows to `inf` for any element at or above **709.7827128933841** and
underflows to `-inf` below **-745.1332191019412**. Below, both boundaries
measured exactly, and the shift that removes them.

## How it works

Softmax, cross-entropy, and log-likelihoods all need the same sub-step: turn
a set of unnormalized log-scores into one normalizing constant, in log space,
without leaving log space in between. Written the way the formula reads —
exponentiate every score, sum, take the log — that middle step is exactly
where it breaks. A `float64` tops out at about `1.8e308`; `exp(x)` reaches
that ceiling once `x` passes roughly 709, so any log-score even moderately
larger than typical (an attention logit before scaling, an unnormalized
mixture weight) sends `exp` to `inf` and `log(inf)` back out as `inf` — a
value that looks like a computed answer but destroys every gradient and
comparison downstream. The mirror failure is quieter: very negative scores
send `exp` to exactly `0.0`, and `log(0)` returns `-inf` even though the true
LSE is an ordinary finite number close to the largest input.

The fix does not change what is computed, only the order the exponents
arrive in. Factor out `e^m` for `m = max(x)`:

```
LSE(x) = log(Σ eˣⁱ) = m + log(Σ e^(xᵢ - m))
```

Every shifted exponent `xᵢ - m` is `<= 0`, so every `e^(xᵢ - m)` lands in
`(0, 1]` — nothing can overflow — and the term where `i` attains the max is
exactly `e^0 = 1`, so the sum is never all-zero either. This is an algebraic
identity, not an approximation: the shift is exact, so a correct
implementation should match a naive one to rounding error everywhere the
naive one survives, and simply keep working where it doesn't.

The same shift-and-recombine shape recurs up the stack. Two-argument
`logaddexp(a, b)` is this trick specialised to `n = 2` and rewritten with
`log1p` for precision near zero — see
[the pairwise version](../tasks/num-stable-log-add-exp-a-b/task.md). One row
`x - LSE(x)` is `log_softmax`, worked through in
[log-softmax via LSE](../tasks/num-log-softmax-via-lse/task.md). And when a
row arrives in chunks too long to hold in memory at once — the case attention
kernels face — the running max and running sum can be updated block by
block and still equal the whole-row answer exactly, which is
[the streaming identity that makes flash attention's backward pass
possible](../tasks/sys-streaming-logsumexp-equals-full-logsumexp/task.md).
It is the same counted-not-timed philosophy as
[memory coalescing](memory-coalescing.md) and
[false sharing](false-sharing.md): the failure is exact and reproducible, so
it should be measured as a number rather than described in words.

## Overflow and underflow measured against input magnitude

A three-element vector `[M, M+1, M-3]` was fed through the naive formula and
the max-shift form for growing `M`, and the naive result was compared against
the shifted one — the true error introduced by skipping the shift.

| max(x) = M | naive log(Σ eˣⁱ) | \|naive − stable\| |
|---|---|---|
| 10 | 11.32656264 | 0.0 |
| 100 | 101.32656264 | 0.0 |
| 710 | **inf** | **inf** |
| 1000 | inf | inf |
| -710 | -708.67343736 | 0.0 |
| -750 | **-inf** | **inf** |
| -1000 | -inf | inf |

Reproduce it:

```bash
pip install mlsys-lab
python3 - <<'PY'
import numpy as np

def naive_lse(x):
    with np.errstate(over="ignore", divide="ignore"):
        return float(np.log(np.sum(np.exp(x))))

def stable_lse(x):
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x)
    return float(m + np.log(np.sum(np.exp(x - m))))

for M in (10, 100, 710, 1000, -710, -750, -1000):
    x = np.array([M, M + 1.0, M - 3.0])
    naive, stable = naive_lse(x), stable_lse(x)
    err = abs(naive - stable) if np.isfinite(naive) else float("inf")
    print(f"M={M:>5}  naive={naive!r:>10}  abs_err={err}")

# exact float64 boundary: smallest x with exp(x) == inf, by bisection
lo, hi = 700.0, 720.0
for _ in range(60):
    mid = (lo + hi) / 2
    if np.isinf(np.exp(mid)):
        hi = mid
    else:
        lo = mid
print("overflow boundary:", hi)
PY
```

The naive column is exact `inf` starting at `M=710`, not merely large — the
boundary bisection in the snippet pins it to `709.7827128933841`, one ULP
above `log(float64 max)`. Below that, `|naive - stable|` is `0.0` to the last
bit: the shift changes nothing about the answer, only which numbers appear as
intermediates. Once `M` crosses either boundary the naive path does not
degrade gracefully — the error is not a growing number, it is `inf`, and
`inf - finite = inf` is where the comparison in this table stops being able
to say anything more precise than "broken."

## Practise it

```bash
mlsys grade num-naive-vs-stable-lse-on-overflow-underflow-fixture
```

[That task](../tasks/num-naive-vs-stable-lse-on-overflow-underflow-fixture/task.md)
gates `rel_err <= 1e-13` against a 50-digit `mpmath` oracle, run on a
pre-built overflow fixture, a pre-built underflow fixture, a mixed-extreme
case, and four ordinary random arrays. The shipped starter raises
`NotImplementedError` and fails outright; a naive `log(sum(exp(x)))` fails
differently — it returns `inf` or `-inf` on the two adversarial fixtures,
which the grader checks with `np.isfinite` before it ever computes an error,
so an answer that is numerically perfect on ordinary inputs still scores
`inf` overall. **Correct-on-the-easy-cases is a failing answer here.**

In roughly increasing difficulty:
[prove shift invariance algebraically](../tasks/alg-prove-logsumexp-shift-invariance/task.md) (no numerics, one identity),
[LogSumExp with max-shift over a chosen axis](../tasks/num-logsumexp-with-max-shift/task.md) (numpy, batched),
[stable two-term logaddexp](../tasks/num-stable-log-add-exp-a-b/task.md) (the `n=2` special case),
[LogSumExp stability across scales in real C++](../tasks/cpu-logsumexp-stability-across-scales/task.md) (compiled, six fixtures), and
[debug a naive log-softmax that underflows](../tasks/llm-debug-naive-log-softmax-underflow/task.md) (find the bug, not write from scratch).

## Common mistakes

- **Shifting by a global constant instead of a per-row max.** On a batch
  `[[5, 6, 4], [800, 801, 799]]`, shifting every row by the batch-wide max
  `801` sends the first row's exponents to `exp(-796)` and beyond — they all
  underflow to `0.0` together, and the row that should evaluate to `6.41`
  comes back `-inf`, while the row that actually contains the max is
  unaffected. Reducing per-row (`axis=-1, keepdims=True`) is what the trick
  requires; a single scalar shift only protects whichever row supplied it.
- **Forgetting to add `m` back.** `log(sum(exp(x - m)))` alone is a valid,
  finite-looking number — it is simply wrong by exactly `m`. For `x = [1000,
  1001, 997]` that is `0.3265626413` instead of `1001.3265626413`, a
  thousand-unit error with no `inf` or `nan` to flag it.
- **Subtracting the max after exponentiating.** `exp(x) - m` instead of
  `exp(x - m)` is a copy-paste away from the correct line and reproduces the
  exact same overflow the trick exists to remove, since `exp(x)` still runs
  on the raw, unshifted values first.
- **Treating the shift as an approximation.** It is an exact algebraic
  identity — the measured table above shows `0.0` absolute error everywhere
  both forms stay finite — so a stable implementation should never be
  expected to trade accuracy for range. If it does, the bug is elsewhere.
- **Checking for `inf`/`nan` after the fact instead of preventing it.**
  By the time `log(0.0)` has produced `-inf`, the information needed to
  recover the true answer is gone; there is no shift that un-underflows a
  sum that already rounded to zero.

## Where else to practise this

From the [full survey of what exists](../LANDSCAPE.md) for this track:

- **[Tensor Puzzles](https://github.com/srush/Tensor-Puzzles)** — 21
  auto-checked broadcasting puzzles, the best-graded thing in this area.
  It does not touch floating-point precision at all, so it is a good
  warm-up for the NumPy mechanics this page assumes and no substitute for it.
- **[Float Exposed](https://float.exposed/)** — flip bits of a `float64` and
  watch the decimal value change live. The best hands-on way to *see* why
  `1.7976931348623157e+308` is the ceiling this page's boundary sits one ULP
  below; it has no exercises, only the visualization.
- **[fp-conv](https://sw23.github.io/fp-conv/)** — the same bit-flipping
  interaction extended to bf16, fp8, and fp4, useful once mixed-precision
  training puts logits in a narrower format than `float64` and the same
  overflow logic applies at a much smaller magnitude.
- **CS231n Assignment 1**'s softmax section has a learner derive a stable
  softmax by hand inside a graded notebook — closely related, narrower in
  scope, and only one assignment rather than a dedicated LSE exercise set.
  The survey's own conclusion for this area is blunt: nowhere else found
  "grades a stable-softmax/log-sum-exp implementation against an
  overflow-triggering input" the way the task above does.

## References

1. Goldberg, D., *What Every Computer Scientist Should Know About
   Floating-Point Arithmetic*, 1991 — the overflow/underflow mechanics this
   page measures. https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html
2. SciPy documentation, `scipy.special.logsumexp` — the independent oracle
   this bank's grader checks answers against.
   https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html
3. Wikipedia, *LogSumExp* — the derivation, the shift-invariance identity,
   and its role as the softmax normalizer. https://en.wikipedia.org/wiki/LogSumExp
