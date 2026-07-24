## Context

In transformer models the key ($K$) and value ($V$) tensors are typically shaped $(n_{\text{seq}}, d)$, where $n_{\text{seq}}$ is the number of tokens in a sequence and $d$ is the hidden dimension.  
When quantizing these tensors we may choose to group‑quantise along **channels** (the feature dimension) or along **tokens** (the sequence dimension).  

For a 2‑D array $X \in \mathbb{R}^{n\times d}$, let

$$
\operatorname{var}_{\text{channel}}(X)=\sum_{j=1}^d \operatorname{Var}(X_{\cdot j}), \qquad
\operatorname{var}_{\text{token}}(X)=\sum_{i=1}^n \operatorname{Var}(X_{i\cdot}),
$$

where $\operatorname{Var}$ is the population variance.  
If we quantize each group to its mean value, the resulting squared‑error equals the corresponding sum of variances.  The axis with the smaller sum therefore yields a lower reconstruction error.

The task is to pick for each of $K$ and $V$ whether channel‑wise or token‑wise grouping gives the lower MSE.

## Task

Implement `pick_kivi_quant_axis(K, V)`:

```python
def pick_kivi_quant_axis(K: np.ndarray, V: np.ndarray) -> tuple[str, str]:
    ...
```

`K` and `V` are 2‑D NumPy arrays of shape `(n, d)`.  
Return a tuple `(k_axis, v_axis)` where each element is either the string `"channel"` or `"token"`, indicating the axis that yields the lower group‑quant MSE for that tensor.

The implementation must use only vectorised NumPy operations; no explicit Python loops are allowed.

## Example

```python
import numpy as np
K = np.array([[0, 0], [10, 10]])          # token variance is zero
V = np.array([[0, 5], [0, 5]])            # channel variance is zero
k_axis, v_axis = pick_kivi_quant_axis(K, V)
print(k_axis, v_axis)   # -> ('token', 'channel')
```

## What the gate checks

The grader computes the exact MSE for both axes on each tensor using NumPy and labels the axis with the lower value.  
It then compares those labels to the tuple returned by your function.  The solution must match exactly; otherwise the `exact_match` gate fails.
