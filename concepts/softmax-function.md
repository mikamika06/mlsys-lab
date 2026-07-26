---
title: "What is softmax function?"
description: "Softmax function explained, with float64 shift invariance, the naive-overflow boundary, and a measured entropy-vs-temperature table you can reproduce, plus a graded exercise."
datePublished: 2026-07-26
dateModified: 2026-07-26
author: Oleksandr Savkov
---

# What is softmax function?

The softmax function turns a vector of real-valued logits into a probability
distribution — one non-negative weight per class, all summing to `1.0`. Computed the way
the formula reads, it returns `nan` once any logit clears **709.7827128933841** in
float64, and shifting every logit by the same constant is supposed to leave the output
untouched exactly. Below: how tightly that shift invariance actually holds in floating
point, where the naive form breaks, and what a temperature sweep from `0.1` to `10` does
to a fixed distribution's entropy.

## How it works

`softmax(x)ᵢ = eˣⁱ / Σⱼ eˣʲ` — exponentiate every logit, then divide each by the total.
Exponentiating makes every output strictly positive and turns additive differences
between logits into multiplicative ratios (`softmax(x)ᵢ / softmax(x)ⱼ = e^(xᵢ − xⱼ)`),
which is why the largest logit always gets the largest share and a logit that is `10`
larger than its neighbor gets roughly `e¹⁰ ≈ 22,026` times the probability mass, not `10`
times. Dividing by the sum turns that pile of positive numbers into a distribution, but
that sum is exactly the kind of accumulation that loses precision when its terms span
many orders of magnitude — the same concern [Kahan summation](kahan-summation.md)
addresses for an ordinary sum, though softmax's fix removes the spread before summing
rather than compensating for it while summing.

That trick rests on one algebraic identity: `softmax(x) = softmax(x − c)` for any
constant `c`, because subtracting `c` from every exponent multiplies numerator and
denominator by the same `e⁻ᶜ`, which cancels. Choosing `c = max(x)` makes every shifted
exponent `≤ 0`, so `e^(xᵢ − c) ∈ (0, 1]` and the sum can never overflow — the identity
measured below, and proved directly in
[the shift-invariance task](../tasks/llm-shift-invariance-proof-softmax-x-softmax-x-c/task.md).
It is the same shift that [log-sum-exp](log-sum-exp.md) applies to its own sum of
exponentials, since softmax's denominator, in log space, *is* log-sum-exp; the two share
one overflow boundary because they share the same `exp` underneath.

Two other axes matter once softmax leaves the whiteboard. Divide every logit by a
temperature `T` before exponentiating and the same shift trick still applies, but the
output distribution gets sharper as `T → 0` and flatter as `T → ∞`, measured below —
the mechanism behind every LLM sampling temperature slider. And softmax is a genuinely
different function from [sigmoid](softmax-vs-sigmoid.md), not a rebranding of it — they
coincide exactly at two classes and diverge from three, which that page measures
directly rather than repeating here. On real hardware, softmax's row-wise max-and-sum is
also where a naive kernel loses parallelism efficiency the same way
[warp divergence](warp-divergence.md) does — a per-row reduction that looks free in
NumPy is a synchronization point across threads in CUDA — and the format the logits are
stored in decides where the overflow boundary actually sits, which is the whole subject
of [bfloat16 vs float16](bfloat16-vs-float16.md).

## Shift invariance, overflow, and temperature — measured

Three things were measured against one stable implementation: how close
`softmax(x)` stays to `softmax(x − c)` in float64 as `|c|` grows, the exact scalar at
which the naive, unshifted form first returns a non-finite value in float64 and in
float32, and how a temperature sweep over a fixed six-logit vector reshapes its entropy
and top probability.

| measurement | value |
|---|---|
| shift invariance, max abs diff, \|c\| ≤ 1,000 (200,000 trials) | 4.180e-13 |
| shift invariance, max abs diff, \|c\| ≤ 1,000,000 (200,000 trials) | 2.126e-11 |
| error growth, larger shift ÷ smaller shift | **50.9×** |
| naive overflow boundary, float64 | **709.7827128933841** |
| naive overflow boundary, float32 | **88.72283554077148** |

| T | entropy | entropy ÷ ln(6) | max probability |
|---|---|---|---|
| 0.1 | 0.000499 | 0.000279 | 0.999955 |
| 0.2 | 0.040732 | 0.022733 | 0.993258 |
| 0.5 | 0.493532 | 0.275445 | 0.859290 |
| 1.0 | 1.155079 | 0.644662 | 0.602904 |
| 2.0 | 1.590703 | 0.887788 | 0.380110 |
| 5.0 | 1.757935 | 0.981122 | 0.243617 |
| 10.0 | 1.783314 | 0.995286 | 0.203213 |

Reproduce it:

```bash
pip install mlsys-lab
python3 - <<'PY'
import numpy as np

def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x)
    e = np.exp(x - m)
    return e / np.sum(e)

def naive_softmax(x, dtype=np.float64):
    x = np.asarray(x, dtype=dtype)
    with np.errstate(over="ignore", invalid="ignore"):
        e = np.exp(x)
        return e / np.sum(e)

# 1. shift invariance: softmax(x) vs softmax(x - c), float64, growing |c|
rng = np.random.default_rng(0)
N = 200_000
errs = {}
for c_max in (1_000, 1_000_000):
    max_diff = 0.0
    for _ in range(N):
        d = rng.integers(2, 12)
        x = rng.uniform(-1e4, 1e4, d)
        c = rng.uniform(-c_max, c_max)
        diff = np.max(np.abs(softmax(x) - softmax(x - c)))
        max_diff = max(max_diff, diff)
    errs[c_max] = max_diff
    print(f"shift invariance, |c| <= {c_max}: max_abs_diff over {N} trials = {max_diff:.3e}")
print(f"error growth ratio (1,000,000 shift / 1,000 shift): {errs[1_000_000] / errs[1_000]:.1f}")

# 2. naive (unshifted) overflow boundary, float64 and float32, by bisection
def overflow_boundary(dtype, lo, hi):
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(80):
            mid = (lo + hi) / 2
            if not np.all(np.isfinite(naive_softmax(np.array([mid, 0.0]), dtype=dtype))):
                hi = mid
            else:
                lo = mid
    return hi

b64 = overflow_boundary(np.float64, 700.0, 720.0)
b32 = overflow_boundary(np.float32, 80.0, 95.0)
print("naive softmax overflow boundary, float64:", b64)
print("naive softmax overflow boundary, float32:", b32)

# 3. temperature sweep: entropy and max probability, fixed 6-logit vector
logits = np.array([2.0, 1.0, 0.1, -1.0, 3.0, 0.5])
max_entropy = np.log(len(logits))
print("max possible entropy for 6 classes, ln(6):", max_entropy)
for T in (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0):
    p = softmax(logits / T)
    ent = -np.sum(p * np.log(p))
    print(f"T={T:>4.1f}  entropy={ent:.6f}  entropy/ln6={ent / max_entropy:.6f}  max_prob={p.max():.6f}")
PY
```

Read the first table as a warning against treating shift invariance as exact just
because the algebra says so: it holds to `4.180e-13` for shifts up to `1,000`, but
letting the shift grow to `1,000,000` costs a full `50.9×` more error — `2.126e-11` —
purely from the catastrophic cancellation in computing `x − c` itself, before softmax's
own internal max-subtraction ever runs. The overflow boundary drops from `709.7827128933841`
in float64 to `88.72283554077148` in float32 — nearly 8× smaller in magnitude — because
float32's exponent field is narrower, the same trade-off [bfloat16 vs float16](bfloat16-vs-float16.md)
measures directly. The temperature table tells the sampling story: at `T=0.1` one class
takes `0.999955` of the mass and entropy is a sliver of its `1.791759469228055` (`ln 6`)
ceiling; by `T=10` entropy has climbed to `0.995286` of that ceiling and the top class
holds only `0.203213` — barely above the uniform `1/6 ≈ 0.1667`.

## Practise it

```bash
mlsys grade llm-shift-invariance-proof-softmax-x-softmax-x-c
```

[That task](../tasks/llm-shift-invariance-proof-softmax-x-softmax-x-c/task.md) gates two
metrics: `accuracy <= 1e-12` against a stable NumPy reference, and
`shift_invariance <= 1e-12` — the exact identity measured above, checked directly on
test vectors including `[1e6, 1e6, 1e6]` and shifts up to `±1000`. The shipped starter
raises `NotImplementedError` and fails both outright. A softmax written without the
max-subtraction — `exp(x) / exp(x).sum()`, no shift at all — passes on small vectors
like `[1, 2, 3]` but fails both gates the moment a test vector crosses `709.7827128933841`:
`exp(1000)` is already `inf`, and `inf / inf` is `nan` no matter what constant you
subtracted first.

More tasks in the same area, roughly increasing in scope:
[stable softmax via max-subtraction](../tasks/alg-naive-softmax-overflow-max-subtraction-fix/task.md)
(`max_abs_err <= 1e-9`, the overflow fix on its own),
[predict which fixtures overflow a naive float32 softmax](../tasks/llm-predict-which-fixtures-overflow-naive-softmax/task.md)
(`exact_match == 1.0`, classification rather than computation — the float32 boundary
measured above is exactly what it gates on),
[softmax survives 300 large logits](../tasks/llm-stable-softmax-survives-300-logits/task.md)
(`max_abs_err < 1e-7` on rows up to `±1000`, the batched, higher-dimensional version),
and [temperature sweep, entropy and KL](../tasks/num-temperature-sweep-entropy-kl/task.md)
(`mean_kl <= 1e-9`, the exact computation behind the second table above).

## Common mistakes

- **Trusting shift invariance to be exact for any constant.** The measured table shows
  `50.9×` more error from a shift of `1,000,000` than from `1,000` — algebraically
  identical, numerically not, because subtracting a huge unrelated constant from a much
  smaller logit loses precision in the subtraction itself, before softmax's own
  max-shift (which always uses `c = max(x)`, never an arbitrary external constant) gets
  a chance to help.
- **Assuming "logits are usually small" makes the naive form safe.** Attention logits
  before scaling, or an unnormalized mixture weight, can exceed the float64 boundary
  `709.7827128933841` or, in float32, the far closer `88.72283554077148` — nearly 8×
  smaller — without warning, silently turning a correct computation into `nan`.
- **Reading a temperature near `0` as "no effect."** At `T=0.1` the measured
  distribution already concentrates `0.999955` of its mass on one class; a temperature
  meant to sharpen a distribution slightly instead collapses it almost to a hard argmax.
- **Confusing softmax with sigmoid past two classes.** They are the same computation
  only at `K=2` — see [softmax vs sigmoid](softmax-vs-sigmoid.md) for the exact point
  and size of the divergence, not restated here.

## Where else to practise this

From the [full survey of what exists](../LANDSCAPE.md) for this track:

- **[CS231n Assignment 1](https://cs231n.github.io/assignments2026/assignment1/)** — a
  learner derives and codes a softmax classifier by hand in raw NumPy, gradient-checked
  in the notebook. Covers the multi-class mechanics; touches neither the shift-invariance
  identity nor either overflow boundary measured here.
- **[Stanford CS336 — Assignment 1](https://github.com/stanford-cs336/assignment1-basics)**
  — builds softmax as one gradable component inside a full from-scratch transformer
  (alongside RMSNorm, RoPE, attention), checked by pytest. Broader scope, narrower depth
  on softmax's own numerics specifically.
- **[Triton-Puzzles](https://github.com/gpu-mode/Triton-Puzzles)** — puzzle #8, "Long
  Softmax," has a learner implement the row-wise max-and-sum pattern in Triton,
  auto-checked against a reference with no GPU required. Kernel-level, not numerics-level.
- **[Float Exposed](https://float.exposed/)** — flip bits of a float64 or float32 by
  hand and watch it become `inf`; the clearest way to *see* why `709.7827128933841` and
  `88.72283554077148` are where each format gives out, with no softmax content itself.
- The landscape survey's own verdict here is blunt: nowhere found "grades a
  stable-softmax/log-sum-exp implementation against an overflow-triggering input" the
  way the tasks above do — this bank's tasks are the only auto-graded coverage of that
  half of the topic.

## References

1. Bishop, C., *Pattern Recognition and Machine Learning*, §4.3.4 — softmax
   (the "normalized exponential") as a probability model over classes.
   https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/
2. Goldberg, D., *What Every Computer Scientist Should Know About Floating-Point
   Arithmetic*, 1991 — the overflow mechanics behind both boundaries measured here.
   https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html
3. Hinton, G., Vinyals, O., Dean, J., *Distilling the Knowledge in a Neural Network*,
   2015 — the temperature-scaled softmax used above, introduced for knowledge
   distillation. https://arxiv.org/abs/1503.02531
