## Context

Converting a pretrained multi-head attention (MHA) checkpoint into a
grouped-query attention (GQA) checkpoint ("uptraining") requires
collapsing each group of $n_{\text{rep}}$ original K (and V) heads into
a single shared head. Every query head in the group then attends against
that one shared key head instead of its own original key head.

For a group of key heads $K_1, \dots, K_{n_{\text{rep}}} \in
\mathbb{R}^{s \times d}$, the induced error from replacing each $K_i$
with a single shared $\hat K$ is measured by the sum-of-squares distance

$$
J(\hat K) = \sum_{i=1}^{n_{\text{rep}}} \lVert K_i - \hat K \rVert_F^2 .
$$

Setting $\nabla_{\hat K} J = 0$ gives $\sum_i (K_i - \hat K) = 0$, i.e.

$$
\hat K^\star = \frac{1}{n_{\text{rep}}} \sum_{i=1}^{n_{\text{rep}}} K_i ,
$$

the **mean** of the group — the unique minimizer of $J$. Any other
choice (e.g. picking one representative head and discarding the rest)
gives strictly higher $J$ whenever the heads in the group actually
differ.

This K-space optimality propagates to the quantity that actually
matters: the attention **logits**. For query $Q_h \in \mathbb{R}^{s_q
\times d}$ of a head $h$ in the group, replacing its own key head $K_h$
with the shared $\hat K$ changes the logits from $Q_h K_h^\top / \sqrt d$
to $Q_h \hat K^\top / \sqrt d$. Summed (or averaged) over the group, the
mean-pooled $\hat K$ keeps this logit drift smaller than picking any
single head does.

## Task

Implement `mean_pool_gqa_logit_mse`:

```python
def mean_pool_gqa_logit_mse(Q: np.ndarray, K: np.ndarray, n_rep: int) -> float:
    ...
```

* `Q` — shape $(n_{\text{heads}}, s_q, d)$, the original per-head queries.
* `K` — shape $(n_{\text{heads}}, s_k, d)$, the original per-head keys of
  one MHA checkpoint.
* `n_rep` — group size; `n_heads` is guaranteed divisible by `n_rep`.
  Heads `[g*n_rep : (g+1)*n_rep)` form group $g$.

For each group, mean-pool its `n_rep` key heads into one shared key head
$\hat K_g = \frac{1}{n_{\text{rep}}}\sum_{i \in g} K_i$. Every head $h$ in
group $g$ then attends with its **own** $Q_h$ against the **shared**
$\hat K_g$ instead of its own $K_h$:

$$
\text{logits}^{\text{orig}}_h = \frac{Q_h K_h^\top}{\sqrt d}, \qquad
\text{logits}^{\text{gqa}}_h = \frac{Q_h \hat K_g^\top}{\sqrt d} .
$$

Return the mean squared error between `logits_gqa` and `logits_orig`
over all heads and all $(q, k)$ positions.

## Example

```python
import numpy as np

Q = np.random.default_rng(0).standard_normal((4, 3, 8))
K = np.random.default_rng(0).standard_normal((4, 5, 8))
n_rep = 2   # heads 0,1 share one pooled key head; heads 2,3 share another

mse = mean_pool_gqa_logit_mse(Q, K, n_rep)
```

## What the gate checks

The gate, **rel_err**, compares your returned MSE against an fp64 NumPy
oracle that mean-pools each group's keys and recomputes the logits the
same way, across several random head/sequence-length/group-size
combinations. The oracle also confirms (via assertion) that this
mean-pool MSE is always strictly lower than the MSE from a "pick one
representative head" baseline — the derivation this task is built on.
Your result must match the mean-pool oracle to a relative error of
`<= 1e-6`; picking a single head instead of averaging the group produces
a clearly larger MSE and fails the gate.
