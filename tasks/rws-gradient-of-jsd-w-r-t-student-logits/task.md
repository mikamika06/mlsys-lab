## Context

Knowledge distillation sometimes trains the student against the
**generalized Jensen–Shannon divergence** instead of plain KL, since JSD
stays finite and well-behaved even when teacher and student put mass on
very different tokens. With mixture weight $\beta \in (0,1)$:

$$
m = \beta p + (1-\beta) q, \qquad
\operatorname{JSD}_\beta(p,q) = \beta\,\mathrm{KL}(p\Vert m) + (1-\beta)\,\mathrm{KL}(q\Vert m),
$$

where $p = \operatorname{softmax}(z_t)$ is the **teacher**'s distribution
(a fixed target — no gradient flows into $z_t$) and
$q = \operatorname{softmax}(z_s)$ is the **student**'s distribution,
which *does* depend on the student's logits $z_s \in \mathbb{R}^V$. To
train the student we need $\nabla_{z_s}\operatorname{JSD}_\beta(p,q)$.

**Step 1 — gradient w.r.t. $q$ (treating $m$'s dependence on $q$
correctly via the product/chain rule).** Differentiating
$\operatorname{JSD}_\beta$ w.r.t. $q_j$ and using $\beta p_j + (1-\beta)
q_j = m_j$ to cancel the terms coming from $m$'s own $q$-dependence
gives the clean closed form

$$
\frac{\partial \operatorname{JSD}_\beta}{\partial q_j} = (1-\beta)\,\log\frac{q_j}{m_j} \;=:\; g_j .
$$

**Step 2 — backprop through softmax.** With $q = \operatorname{softmax}(z_s)$,
the softmax Jacobian is $\partial q_j/\partial z_{s,k} = q_j(\delta_{jk} -
q_k)$, so

$$
\frac{\partial \operatorname{JSD}_\beta}{\partial z_{s,k}}
= \sum_j g_j\, q_j(\delta_{jk}-q_k)
= q_k\Bigl(g_k - \sum_j q_j g_j\Bigr).
$$

## Task

Implement `jsd_grad_wrt_student_logits(teacher_logits, student_logits, beta)`:

```python

def jsd_grad_wrt_student_logits(teacher_logits: list[float], student_logits: list[float], beta: float) -> list[float]:
    ...
```

- `teacher_logits`, `student_logits`: 1-D `float64` arrays of shape
  `(V,)`, raw (unnormalized) logits.
- `beta`: Python float in `(0, 1)`.

Return a 1-D `float64` array of shape `(V,)`: the gradient of
$\operatorname{JSD}_\beta(\operatorname{softmax}(z_t),
\operatorname{softmax}(z_s))$ with respect to `student_logits`, holding
`teacher_logits` fixed (i.e. no gradient is taken w.r.t. `teacher_logits`).

## Example

```python
teacher_logits = [2.0, 0.0, -1.0]
student_logits = [0.5, 0.5, 0.0]
grad = jsd_grad_wrt_student_logits(teacher_logits, student_logits, beta=0.5)
# grad has shape (3,) and sums to (very close to) 0 -- softmax-Jacobian
# outputs are always in the tangent space of the simplex.
```

## What the gate checks

The grader builds an oracle gradient with **central finite differences**
directly on $f(z_s) = \operatorname{JSD}_\beta(\operatorname{softmax}(z_t),
\operatorname{softmax}(z_s))$ — perturbing each logit by $\pm\epsilon$
independently and never using the closed-form expression above — across
several random `(teacher_logits, student_logits, beta)` triples with
`V` between 5 and 10.

The gate metric is `max_abs_err`, the largest absolute
elementwise difference between your gradient and the finite-difference
oracle across all cases; it must be `< 1e-5`. Forgetting the softmax
Jacobian's $-q_k \sum_j q_j g_j$ correction term, differentiating w.r.t.
the wrong logits, or missing the $(1-\beta)$ factor will all produce a
gradient that diverges from the finite-difference oracle by far more than
that threshold.
