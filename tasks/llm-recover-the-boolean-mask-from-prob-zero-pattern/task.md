## Context

In many transformer architectures, attention scores are masked before the
softmax.  A mask $M \in \{0, 1\}^{n \times n}$ is a binary matrix indicating
which pairs of positions are allowed (1) or forbidden (0).  The attention
probability matrix $P \in \mathbb{R}^{n \times n}$ is obtained by masking the
logits and then applying softmax:

$$
P = \mathrm{softmax}(L \oplus (1 - M) \cdot (-\infty)),
$$

where $\oplus$ is element‑wise addition.  Equivalently, any entry $P_{ij}$ for
which $M_{ij} = 0$ must be exactly $0$ (because the corresponding logit is
replaced by $-\infty$), while entries where $M_{ij} = 1$ must be strictly
positive (because the sum of all allowed entries is 1 and no other non‑zero
probability exists).

Thus, given the probability matrix $P$, we can recover the original mask by
checking which entries are zero and which are non‑zero:

$$
M_{ij} = \begin{cases}
1 & \text{if } P_{ij} > 0,\\
0 & \text{if } P_{ij} = 0.
\end{cases}
$$

In practice, floating‑point arithmetic may produce very small but non‑zero
values where zeros are expected.  A robust recovery uses a tolerance
$\varepsilon > 0$: treat entries with $P_{ij} \le \varepsilon$ as zero.

## Task

Implement `recover_mask(P, eps=1e-12)`:

```python
def recover_mask(P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    ...
```

It takes an attention probability matrix $P$ of shape $(n, n)$ and a tolerance
$\varepsilon$, and returns a boolean matrix $M$ of the same shape, where
$M_{ij}$ is True iff $P_{ij} > \varepsilon$.

## Example

```python
import numpy as np

P = np.array([
    [0.0, 0.5, 0.0],
    [0.3, 0.0, 0.7],
    [0.0, 0.0, 1.0]
])

M = recover_mask(P, eps=1e-12)
# Expected:
# [[False,  True, False],
#  [ True, False,  True],
#  [False, False,  True]]
```

## What the gate checks

Exact match of the recovered mask against the true mask computed from $P$ using
the same $\varepsilon$.  The grader:

1. Generates a synthetic causal mask and a random set of logits.
2. Manually masks the logits with $-\infty$ where the mask is 0.
3. Computes the softmax to obtain $P$.
4. Calls your `recover_mask(P, eps=1e-12)` and compares the result to the
   original mask entry‑wise.

Every entry must agree (score $1.0$).  Any mismatch gives $0.0$.
