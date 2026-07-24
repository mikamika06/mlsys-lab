## Context

Memory-efficient attention (FlashAttention-style) never keeps the full
$n\times n$ probability matrix $P$ around for the backward pass. Instead
the forward pass saves only the output $O$ and two small per-row
statistics, the row max $m$ and row softmax-normalizer $l$:

$$
S = \frac{QK^\top}{\sqrt d} + B, \qquad
m_i = \max_j S_{ij}, \qquad
l_i = \sum_j \exp(S_{ij} - m_i)
$$

where $B \in \mathbb{R}^{n\times n}$ is a fixed **additive bias** applied
to the scaled scores before softmax — the same slot used by ALiBi
positional bias or an additive attention mask. $B$ has no gradient of its
own in this exercise; it only shapes $S$.

On the backward pass, $P$ is **recomputed** on the fly, from $Q$, $K$,
$B$ and the saved $(m, l)$ — never from a stored matrix:

$$
P_{ij} = \exp(S_{ij} - m_i) \big/ l_i
$$

Given the upstream gradient $dO = \partial L/\partial O$, the standard
softmax-attention backward recurrence applies unchanged, because $B$ is
*additive* to $S$ and therefore contributes nothing extra to
$\partial S/\partial Q$ or $\partial S/\partial K$:

$$
dV = P^\top dO, \qquad
dP = dO\,V^\top, \qquad
D_i = \sum_j dP_{ij} P_{ij}
$$

$$
dS = P \odot (dP - D), \qquad
dQ = \frac{dS\,K}{\sqrt d}, \qquad
dK = \frac{dS^\top Q}{\sqrt d}
$$

## Task

Implement `biased_flash_backward`:

```python
def biased_flash_backward(Q, K, V, B, dO, m, l):
    ...
```

- `Q`, `K`, `V`: `(n, d)`.
- `B`: `(n, n)`, the additive bias used on the forward pass.
- `dO`: `(n, d)`, upstream gradient w.r.t. the forward output.
- `m`, `l`: `(n,)`, the row max and row softmax-normalizer saved from the
  forward pass — i.e. computed from `S = Q@K.T/sqrt(d) + B` exactly as
  above.
- Recompute `P` from `Q`, `K`, `B`, `m`, `l` (do **not** assume any other
  cached matrix is available) and return `(dQ, dK, dV)`, each shaped like
  `Q`, `K`, `V` respectively. `B` itself needs no gradient.

## Example

```python
import numpy as np

rng = np.random.default_rng(0)
Q = rng.standard_normal((3, 4))
K = rng.standard_normal((3, 4))
V = rng.standard_normal((3, 4))
B = rng.standard_normal((3, 3)) * 0.5
dO = rng.standard_normal((3, 4))

d = Q.shape[1]
S = Q @ K.T / np.sqrt(d) + B
m = S.max(axis=1)
l = np.exp(S - m[:, None]).sum(axis=1)

dQ, dK, dV = biased_flash_backward(Q, K, V, B, dO, m, l)
```

## What the gate checks

The grader builds one small seeded `(Q, K, V, B, dO)` case, computes `m`
and `l` from it, and compares your `(dQ, dK, dV)` against **central
finite differences** of the scalar loss
$L = \sum (\operatorname{softmax}(QK^\top/\sqrt d + B)\,V) \odot dO$
with respect to each entry of `Q`, `K`, `V` independently — a numeric
oracle that never calls your function and never hardcodes an expected
gradient.

`max_abs_err` is the worst per-entry max-abs-error across `dQ`, `dK`,
`dV` and must be `<= 1e-4`. Dropping the row-sum correction term `D`,
forgetting the `1/sqrt(d)` scale on `dQ`/`dK`, transposing `dS` in the
wrong place, or recomputing `P` without the bias `B` all produce a
gradient error far above this threshold.
