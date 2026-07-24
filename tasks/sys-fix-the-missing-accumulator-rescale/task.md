## Context

The softmax function maps a vector of real scores $x \in \mathbb{R}^n$ to a probability distribution
$$p_i = \frac{\exp(x_i)}{\sum_{j=1}^{n}\exp(x_j)}.$$
When the entries of $x$ are large, computing $\exp(x_i)$ directly can overflow.  
A common streaming (online) implementation keeps two running values:

* $m$, the maximum score seen so far,
* $S = \sum_k \exp(s_k - m)$, a sum of exponentials *relative* to that maximum.

When a new score $s$ arrives we update

1. If $s > m$ then all previously accumulated terms must be rescaled:
   $$S \leftarrow S \cdot \exp(m - s),$$
   and the new maximum becomes $m \leftarrow s$.
2. Add the contribution of the new score:
   $$S \leftarrow S + \exp(s - m).$$

After processing a batch of scores, the probability for each element is
$$p_i = \frac{\exp(x_i - m)}{S}.$$

If step 1 (the rescaling) is omitted, the accumulator $S$ grows too large whenever a new maximum appears, producing probabilities that are far from correct.

## Task

Implement the function

```python
def streaming_softmax(scores: np.ndarray, acc=None):
    ...
```

* `scores` – a 1‑D NumPy array of floats.
* `acc` – either `None` or a tuple `(m, S)` representing the current running maximum and sum.  
  If `None`, start from scratch.

The function must return a tuple `(probs, (new_m, new_S))` where

* `probs` is a NumPy array of shape `(len(scores),)` containing the softmax probabilities for this batch,
* `(new_m, new_S)` are the updated running maximum and sum that can be passed to a subsequent call.

Use only NumPy; no external libraries.  The implementation must correctly perform the rescaling step described above.

## Example

```python
import numpy as np

scores = np.array([0.0, 1000.0])
probs, acc = streaming_softmax(scores)
print(probs)          # [0.0, 1.0] (within floating‑point limits)

# Subsequent call with a new batch:
more_scores = np.array([-500.0, 2000.0])
probs2, acc = streaming_softmax(more_scores, acc)
```

## What the gate checks

The grader computes the reference softmax for each test array using
`np.exp(scores) / np.sum(np.exp(scores))`.  
It then evaluates the global relative L2 error
$$\mathrm{rel\_err} = \frac{\|p_{\text{cand}}-p_{\text{ref}}\|}{\|p_{\text{ref}}\|}.$$
The solution must achieve $\mathrm{rel\_err}\le 10^{-5}$ on a set of adversarial arrays with large score spreads.  
A missing rescaling step will produce errors far above this threshold and fail the gate.
