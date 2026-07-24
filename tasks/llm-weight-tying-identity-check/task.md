## Context

Most decoder-only language models **tie** their input embedding matrix and their
output ("LM head") matrix: the *same* weight matrix
$E \in \mathbb{R}^{V \times d}$ ($V$ = vocabulary size, $d$ = hidden size) is used
both to look up token vectors on the way in and to project the hidden state to
vocabulary logits on the way out.

Two operations define the round trip for a single token id $t$:

- **Embedding lookup.** Selecting row $t$ is exactly multiplying a one-hot row
  vector $\mathbf{1}_t \in \mathbb{R}^{V}$ by $E$:
  $$h = \mathbf{1}_t\, E = E_t \in \mathbb{R}^{d}.$$
- **Tied LM head.** With weight tying the head has no separate matrix; the logits
  are the hidden vector projected back through $E$:
  $$\text{logits} = h\, E^\top \in \mathbb{R}^{V}.$$

Composing the two, the logits produced for token $t$ are
$$\mathbf{1}_t\, E\, E^\top = E_t\, E^\top = (E E^\top)_t,$$

i.e. **row $t$ of the Gram matrix $E E^\top$**. Stacking every token
($t = 0, \dots, V-1$) therefore reconstructs the full symmetric Gram matrix:
$$\text{full logits} = E E^\top .$$

This is the *weight-tying identity*: the entire embed-then-unembed pipeline
through a tied head collapses to $E E^\top$, and its $(i, j)$ entry is just the
dot product $\langle E_i, E_j \rangle$.

## Task

Implement `tied_identity_logits`:

```python
def tied_identity_logits(E: np.ndarray, token_ids: np.ndarray) -> np.ndarray:
    ...
```

Given the tied weight matrix `E` of shape $(V, d)$ and a 1-D array `token_ids` of
length $n$, return the logits of shape $(n, V)$ that the tied model produces:
embed each token id (row gather / one-hot lookup) and then apply the tied LM head
($h\,E^\top$). Return a `float64` array. Vectorize with NumPy — do not loop over
tokens in Python.

## Example

```python
import numpy as np
E = np.array([[1.0, 0.0],
              [0.0, 2.0],
              [1.0, 1.0]])          # V=3, d=2
logits = tied_identity_logits(E, np.array([0, 2]))
# row for token 0 -> <E0,E0>, <E0,E1>, <E0,E2>
# row for token 2 -> <E2,E0>, <E2,E1>, <E2,E2>
# [[1. 0. 1.]
#  [1. 2. 2.]]

# Feeding every token id reconstructs the Gram matrix E @ E.T:
full = tied_identity_logits(E, np.arange(3))
# [[1. 0. 1.]
#  [0. 4. 2.]
#  [1. 2. 2.]]
```

## What the gate checks

Two gates, each compared against an independent NumPy oracle (nothing is
hardcoded):

1. **max_abs_err** — Across several random $(V, d)$ matrices and random
   `token_ids`, the worst-case $\lVert \cdot \rVert_\infty$ difference between
   your output and the oracle $\big(\text{one\_hot}(\text{token\_ids})\, E\big)\,E^\top$
   must be $\le 10^{-6}$.

2. **identity_err** — For `token_ids = np.arange(V)`, your output must match the
   Gram matrix $E E^\top$ to $\le 10^{-6}$, directly demonstrating the
   weight-tying identity.

The oracle uses $d \neq V$, so a solution that forgets the head transpose or
returns only the gathered embeddings has the wrong shape and fails.
