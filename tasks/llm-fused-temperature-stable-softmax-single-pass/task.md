## Context

A **temperature-scaled softmax** turns a row of logits $x \in \mathbb{R}^n$ into a
probability distribution, where the temperature $T > 0$ controls sharpness
($T < 1$ sharpens, $T > 1$ smooths):

$$
p_i = \frac{e^{x_i / T}}{\sum_j e^{x_j / T}} .
$$

Evaluated directly this overflows: with large logits — or a small $T$ that
amplifies them — $e^{x_i / T}$ explodes to $\infty$. The standard fix subtracts the
maximum $m = \max_j (x_j / T)$ before exponentiating, which leaves the result
unchanged but keeps every exponent $\le 0$:

$$
p_i = \frac{e^{x_i / T - m}}{\sum_j e^{x_j / T - m}} .
$$

The textbook implementation makes **three passes** over the row: one to find $m$,
one to accumulate $\sum_j e^{x_j/T-m}$, and one to divide. FlashAttention-style
kernels instead fold the first two into a **single online pass**. Keep a running
max $m$ and a running normalizer $d$; when a new element $z = x_i / T$ raises the
max from $m_{\text{old}}$ to $m_{\text{new}}$, rescale the accumulator so it stays
exact without revisiting earlier elements:

$$
d \leftarrow d \cdot e^{\,m_{\text{old}} - m_{\text{new}}} + e^{\,z - m_{\text{new}}}.
$$

After one sweep, $m$ is the global max and $d = \sum_j e^{x_j/T-m}$, so the row can
be normalized in one vectorized step.

## Task

Implement `fused_softmax(logits, T)`:

```python
def fused_softmax(logits, T):
    ...
```

It takes a 1-D array-like of logits and a scalar temperature `T > 0`, and returns
the temperature-scaled softmax as a `float64` NumPy array of the same length.

- Fuse the temperature division into the reduction and compute the running max and
  running normalizer in **one** pass over the row (the online-softmax update above).
- Stay numerically stable: never exponentiate the raw scaled logits.
- Do **not** loop over the row more than once in Python — do the final
  normalization with a single vectorized NumPy expression.

## Example

```python
import numpy as np
logits = np.array([30.0, 32.0, 35.0])
p = fused_softmax(logits, T=0.1)   # scaled logits reach 350 -> naive exp overflows
# [1.92874985e-22, 4.24835426e-14, 1.00000000e+00]
print(p.sum())   # 1.0
```

## What the gate checks

Two gates:

- **max_abs_err** — the largest absolute difference $\max_i |p_i - p_i^{\text{oracle}}|$
  against a NumPy float64 stable-softmax oracle, over cases that include raw
  overflow, small-$T$ amplified overflow, ties, and large negatives. Must be
  $< 10^{-6}$. A solution that exponentiates unshifted logits overflows to
  `inf`/`nan` and fails here.

- **line_events** — using `sys.settrace`, the grader counts Python line events
  during one call on a fixed 256-element row. A single online pass emits roughly
  $5n$ events; a naive max / sum / divide triple loop roughly triples that. The
  count must stay $\le 1700$, which only a genuinely single-pass implementation
  meets (a fully vectorized NumPy reduction passes too, since its loop runs in C).
