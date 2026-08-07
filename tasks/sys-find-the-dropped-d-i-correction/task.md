## Context

FlashAttention's backward pass never materialises the full $n \times n$
attention matrix. Instead, the forward pass saves only the output $O$ and a
per-row **logsumexp statistic** $L_i = m_i + \log \sum_j \exp(S_{ij}-m_i)$
(where $m_i$ is the row max used for numerically-stable softmax). During the
backward pass, the attention weights are **recomputed** cheaply from $L$
instead of being read back from memory:
$$
S_{ij} = \text{scale} \cdot (Q_i \cdot K_j), \qquad
P_{ij} = \exp(S_{ij} - L_i) \; \big(= \operatorname{softmax}_j(S_{ij})\big).
$$

Given the upstream gradient $dO$, the value-gradient is the easy part —
softmax weights are just a linear combination weight for $V$:
$$
dV = P^\top dO .
$$

The gradient through the softmax itself is where a classic bug lives.
Writing $dP = dO\, V^\top$, the *naive* guess $dS_{ij} = P_{ij}\, dP_{ij}$ is
**wrong**, because softmax rows are normalised — increasing $S_{ij}$ doesn't
just increase $P_{ij}$, it also decreases every other $P_{ik}$ in the same
row. The correct softmax-Jacobian-vector-product needs a per-row correction
term
$$
D_i = \sum_{d} dO_{i,d}\, O_{i,d} \qquad \big(\text{i.e. } D = \operatorname{rowsum}(dO \odot O)\big),
$$
which is algebraically equal to $\operatorname{rowsum}(P \odot dP)$ but is
far cheaper to compute (an $O(n\,d)$ dot product instead of an $O(n^2)$
row-sum), since $O$ and $dO$ are already resident. With this correction,
$$
dS_{ij} = P_{ij}\,\big(dP_{ij} - D_i\big),
$$
and the remaining gradients follow from the linear score map:
$$
dQ = \text{scale} \cdot (dS\, K), \qquad dK = \text{scale} \cdot (dS^\top Q).
$$

The implementation below computes `dV` and `dP` correctly, recomputes `P`
from the saved `L` correctly, but **drops the $D_i$ correction** when
forming `dS` — it uses `dS = P * dP` instead of `dS = P * (dP - D_i)`. This
silently produces the wrong `dQ` and `dK` (though, notice, the correct
`dV`) for any input where a row's `P` isn't one-hot.

## Task

Fix `flash_attention_backward` so it restores the missing $D_i$ correction
term:

```python
def flash_attention_backward(Q: list[list[float]], K: list[list[float]], V: list[list[float]], O: list[list[float]], L: list[float], dO: list[list[float]], scale: float):
    ...
```

* `Q, K, V` — float arrays of shape $(n, d)$ (single head, single sequence).
* `O` — float array of shape $(n, d)$, the forward output
  $O = \operatorname{softmax}(\text{scale}\cdot QK^\top)\,V$.
* `L` — float array of shape $(n,)$, the row logsumexp saved by the forward
  pass, so that $P_{ij} = \exp(S_{ij} - L_i)$.
* `dO` — float array of shape $(n, d)$, upstream gradient w.r.t. $O$.
* `scale` — the attention score scale factor (e.g. $1/\sqrt{d}$).

Return `(dQ, dK, dV)`, each shaped like `Q`, `K`, `V` respectively, computed
using the recompute-then-correct formulas above.

## Example

For a single row where `P` is one-hot (attention fully concentrated on one
key), $D_i$ happens to equal $dP$ at that key, making the missing-term bug
invisible — which is exactly why it isn't caught by a quick sanity check.
For any row with a spread-out (non one-hot) softmax, the buggy `dS` and the
correct `dS` diverge, and so do the resulting `dQ`, `dK`.

## What the gate checks

A single gate, **max_abs_err**, builds several small seeded random
`(Q, K, V, dO)` instances, runs the real forward pass to get
`O` and `L`, and computes the ground-truth `dQ, dK, dV` via **central finite
differences** on the scalar loss $\ell(Q,K,V) = \sum (O(Q,K,V) \odot dO)$ —
an independent numerical oracle that never calls your backward code. Your
function's `dQ, dK, dV` are compared element-wise to these numerical
gradients; the worst-case max absolute error over all trials must be
$\le 10^{-4}$. Any exception or wrong output shape counts as a failing
(`1e9`) error.
