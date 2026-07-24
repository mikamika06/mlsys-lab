## Context

DistilBERT's distillation objective sums three terms: a soft
cross-entropy on the teacher's logits, the student's own supervised
loss, and a **cosine embedding loss** that pulls the student's hidden
states toward the teacher's *direction* (not magnitude) in embedding
space. For teacher/student hidden-state vectors $h_t, h_s
\in\mathbb{R}^d$ (one pair per batch row), with target $=1$ (they
should align):

$$
\text{loss}_i = 1 - \cos(h_{t,i}, h_{s,i}) = 1 - \frac{h_{t,i}\cdot h_{s,i}}{\lVert h_{t,i}\rVert\, \lVert h_{s,i}\rVert + \varepsilon}
$$

$$
\mathcal{L} = \frac{1}{B}\sum_{i=1}^{B} \text{loss}_i
$$

Since the teacher is frozen, only $\partial\mathcal{L}/\partial h_s$
matters for training. Writing $a=h_{t,i}$, $b=h_{s,i}$,
$n_a=\lVert a\rVert$, $n_b=\lVert b\rVert$, $d = n_a n_b + \varepsilon$:

$$
\frac{\partial \text{loss}_i}{\partial b} =
-\left(\frac{a}{d} - \frac{(a\cdot b)\, n_a}{n_b\, d^2}\, b\right)
$$

$$
\frac{\partial \mathcal{L}}{\partial h_s} = \frac{1}{B}\cdot\frac{\partial \text{loss}_i}{\partial b}\Big|_{\text{row } i}
$$

(each row's gradient only depends on that row — the batch mean just
divides every row's per-example gradient by $B$).

## Task

Implement `cosine_embedding_loss_and_grad`:

```python
def cosine_embedding_loss_and_grad(h_t: np.ndarray, h_s: np.ndarray, eps: float = 1e-8):
    ...
```

- `h_t`: `(B, d)` `float64` teacher hidden states (treated as constant).
- `h_s`: `(B, d)` `float64` student hidden states (the variable we differentiate w.r.t.).
- `eps`: small constant added to the denominator for numerical stability.

1. For each row $i$: `na = ||h_t[i]||`, `nb = ||h_s[i]||`,
   `dot = h_t[i] . h_s[i]`, `denom = na*nb + eps`,
   `cos = dot/denom`, `loss_i = 1 - cos`.
2. `loss = mean(loss_i)` over the batch.
3. `grad[i] = -(h_t[i]/denom - (dot*na/(nb*denom**2)) * h_s[i]) / B`
   (the formula above, already divided by the batch size `B`).

Return `(loss, grad)`: `loss` a Python `float`, `grad` a `(B, d)`
`float64` array (`dL/d h_s`).

## Example

```python
import numpy as np
h_t = np.array([[1.0, 0.0]])
h_s = np.array([[0.0, 2.0]])   # orthogonal to h_t
loss, grad = cosine_embedding_loss_and_grad(h_t, h_s)
# cos(h_t, h_s) == 0, so loss ~= 1.0 (maximum misalignment penalty)
```

## What the gate checks

The grader builds several seeded `(h_t, h_s)` batches and computes the
reference loss with the exact formula above, and the reference
gradient two ways: the same closed form, **and** an independent
central-finite-difference estimate of `d(mean loss)/d(h_s[i,j])` for
every entry (perturbing each `h_s[i,j]` by `±1e-6` and re-evaluating
the loss) — the two must already agree to high precision before the
grader trusts either as ground truth.

`loss_rel_err` is the relative error between your returned `loss` and
the oracle's (must be `<= 1e-6`) — catches a wrong cosine formula or
sum vs. mean. `grad_fd_rel_err` is the relative L2 error between your
`grad` and the finite-difference estimate (must be `<= 1e-5`) — this
independently exercises the actual chain rule, so it catches a wrong
derivative (e.g. differentiating w.r.t. `h_t` instead of `h_s`,
dropping the product-rule term from `nb`, or forgetting the `1/B`
batch-mean factor) even when the forward loss is correct.
