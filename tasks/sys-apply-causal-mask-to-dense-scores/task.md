## Context

Scaled dot-product attention computes softmax over a score matrix $S \in \mathbb{R}^{n \times n}$.
In causal (auto-regressive) attention, token $i$ may attend only to tokens $j \le i$.
This is enforced by a **causal mask**: set every entry above the diagonal to $-\infty$:

$$\hat S_{ij} = \begin{cases} S_{ij} & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}$$

Because $\exp(-\infty) = 0$, the softmax $\text{softmax}(\hat S)_i$ places zero probability
on every future token. Masking is applied **pre-softmax** — modifying scores, not
post-softmax probabilities — which is the standard formulation in transformer implementations.

## Task

Implement `apply_causal_mask`:

```python
import numpy as np

def apply_causal_mask(S: np.ndarray) -> np.ndarray:
    """Return a copy of S with all entries above the main diagonal set to -inf."""
    ...
```

The input is an $(n, n)$ score matrix (any dtype). Return a **new** `float64` array of the
same shape. Entries on or below the main diagonal ($j \le i$) must be unchanged. All
entries above the main diagonal ($j > i$) must be $-\infty$. Do not modify the input array.

## Example

```python
S = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0],
              [7.0, 8.0, 9.0]])
result = apply_causal_mask(S)
# result:
# [[  1.  -inf -inf]
#  [  4.   5.  -inf]
#  [  7.   8.   9. ]]

# Verify pre-softmax masking produces a valid probability distribution:
probs = np.exp(result) / np.exp(result).sum(axis=-1, keepdims=True)
# probs[0] = [1.0, 0.0, 0.0]   (token 0 attends only to itself)
# probs[1] = [0.5, 0.5, 0.0]   (token 1 attends to tokens 0, 1)
# probs[2] = [0.333.., 0.333.., 0.333..]  (token 2 attends to all three)
```

## What the gate checks

One gate: **max_abs_err** < $10^{-6}$.

The grader computes a NumPy reference: copy $S$ to `float64`, set upper-triangle
entries to $-\infty$, and compare element-wise. Any deviation above the threshold fails.

The reference also validates the output is a valid pre-softmax mask by confirming
$\text{softmax}(\text{output})_{ij} = 0$ for all $j > i$ (strictly, max entry above
diagonal < $10^{-6}$). A post-softmax hack that zeros probabilities but does not set
scores to $-\infty$ would not match the reference score matrix and fail.
