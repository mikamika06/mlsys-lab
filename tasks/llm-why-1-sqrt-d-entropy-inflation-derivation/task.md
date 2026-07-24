## Context

Scaled dot-product attention compares query and key vectors using dot products. For
queries $Q \in \mathbb{R}^{n \times d}$ and keys $K \in \mathbb{R}^{n \times d}$,
the score matrix is

$$
S = QK^\top .
$$

A row of attention probabilities is computed with softmax:

$$
p_i = \frac{\exp(s_i)}{\sum_j \exp(s_j)} .
$$

Without scaling, the variance of dot products grows with the dimension $d$. If each
component of $q$ and $k$ has variance $1$, then

$$
\mathrm{Var}(q^\top k) = d .
$$

Large score magnitudes make softmax concentrate on a few positions, reducing Shannon
entropy:

$$
H(p) = -\sum_i p_i \log(p_i).
$$

The standard scaled attention uses

$$
S_{\mathrm{scaled}} = \frac{QK^\top}{\sqrt{d}},
$$

which keeps score variance approximately constant as $d$ changes.

## Task

Implement `entropy_inflation_ratio(Q, K)`:

```python
def entropy_inflation_ratio(Q: np.ndarray, K: np.ndarray) -> float:
    ...
```

The function receives two NumPy arrays with shapes $(n, d)$. Compute the mean row
entropy of the unscaled attention distribution and the mean row entropy of the
scaled attention distribution. Return the ratio

$$
R = \frac{\mathrm{mean}(H(\mathrm{softmax}(QK^\top / \sqrt{d})))}{\mathrm{mean}(H(\mathrm{softmax}(QK^\top)))} .
$$

Use NumPy operations. The result must be a Python `float`.

## Example

```python
import numpy as np

Q = np.array([[1., 0.], [0., 1.]])
K = np.array([[1., 0.], [0., 1.]])

ratio = entropy_inflation_ratio(Q, K)
# ratio is greater than 1 because scaling increases entropy
```

## What the gate checks

The gate builds deterministic query and key matrices and computes the reference value
using a NumPy softmax and entropy implementation. The submitted function is compared
with the oracle ratio using relative error:

$$
\mathrm{rel\_err} =
\frac{\lVert r_{\mathrm{candidate}} - r_{\mathrm{reference}}\rVert}
{\lvert r_{\mathrm{reference}}\rvert + 10^{-12}} .
$$

The error must satisfy $\mathrm{rel\_err} < 10^{-3}$.
