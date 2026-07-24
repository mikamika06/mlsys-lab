## Context

In attention, a query row attends to a set of keys. Padding, causal, and
sliding-window masks all work the same way: masked-out key positions are pushed
to $-\infty$ before the softmax so they receive zero probability. For a row of
logits $s \in \mathbb{R}^{m}$ with a boolean keep-mask $k \in \{0,1\}^m$
(True = attend to this key, False = masked out), the masked softmax is

$$w_j = \frac{\exp(\tilde{s}_j - \max_l \tilde{s}_l)}{\sum_{l} \exp(\tilde{s}_l - \max_l \tilde{s}_l)}, \qquad
\tilde{s}_j = \begin{cases} s_j & k_j = \text{True} \\ -\infty & k_j = \text{False}. \end{cases}$$

This is stable as long as at least one key is kept. But when a **query is fully
padded** — every key in its row is masked out — the whole row is $-\infty$. Then
the row max is $-\infty$, the shift $\tilde{s}_j - \max_l \tilde{s}_l$ becomes
$-\infty - (-\infty) = \text{NaN}$, and the normalisation degenerates to
$0 / 0 = \text{NaN}$. A single such row leaks NaNs into the whole attention
output and, through the residual stream, into the loss.

The safe-softmax fix detects rows whose denominator is $0$ (equivalently, rows
with no kept key) and defines their output to be the zero vector instead of
NaN — a fully padded query simply contributes nothing.

## Task

Fix `masked_softmax` in `starter.py`:

```python
def masked_softmax(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ...
```

`scores` is a 2-D `float64` array of shape $(n, m)$ (one row per query).
`mask` is a boolean array of the same shape where `True` marks a key that
should be attended to and `False` marks a masked-out (padded) key. Return the
$(n, m)$ array of attention weights where:

* each row that keeps at least one key is the softmax over exactly its kept
  entries (masked positions get weight $0$ and the kept weights sum to $1$),
* each **fully masked** row (no kept key) is the all-zeros vector — no NaN,
  no inf.

The starter builds the $-\infty$ logits and does a standard stable softmax,
which is correct for normal rows but produces NaN for fully padded rows. Make
the output finite for every row.

## Example

```python
import numpy as np

scores = np.array([[1.0, 2.0],
                   [0.5, 1.5]])
mask = np.array([[True, False],    # keeps only key 0
                 [False, False]])  # fully padded query
masked_softmax(scores, mask)
# [[1. 0.]     row 0: softmax over the single kept key -> 1.0 there
#  [0. 0.]]    row 1: fully masked -> all zeros (not NaN)
```

## What the gate checks

One gate — **max_abs_err** must be $< 10^{-6}$. The grader compares your output
against an independent per-row NumPy oracle (softmax over each row's kept keys,
zeros for fully padded rows) over several cases that all include at least one
fully masked query. If any returned value is NaN or inf, or the shape is wrong,
the error is reported as infinite and the gate fails.
