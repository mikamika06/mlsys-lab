## Context

`LLM.int8()` mixed-precision matrix multiplication (bitsandbytes) keeps
**outlier feature columns** — hidden-state channels that occasionally take
on very large magnitudes — in fp16, and int8-quantizes everything else. A
column $j$ of an activation matrix $X \in \mathbb{R}^{n \times d}$ (rows =
tokens, columns = hidden features) is flagged as an outlier column if its
absolute maximum over all tokens meets or exceeds a fixed threshold
(the paper's default is $6.0$):

$$
\text{is\_outlier}(j) \iff \max_{i=1,\dots,n} \left| X_{i,j} \right| \;\ge\; \tau, \qquad \tau = 6.0
$$

## Task

Implement `detect_outlier_columns`:

```python
def detect_outlier_columns(X: np.ndarray, threshold: float = 6.0) -> np.ndarray:
    ...
```

* `X` — 2-D float array of shape `(n_tokens, hidden_dim)`.
* `threshold` — the outlier threshold $\tau$ (default `6.0`).

Return a 1-D integer NumPy array of the **sorted, unique** column indices
$j$ satisfying $\max_i |X_{i,j}| \ge \tau$ (empty array if none qualify).

## Example

```python
import numpy as np
X = np.array([
    [0.1, 7.5, -0.2],
    [0.3, -1.0, 6.0],
    [-0.2, 0.4, 0.1],
])
detect_outlier_columns(X, threshold=6.0)
# -> array([1, 2])   (column 0 max|.|=0.3 < 6.0; column 1 max|.|=7.5 >= 6.0; column 2 max|.|=6.0 >= 6.0)
```

## What the gate checks

Gate **exact_match** builds random token x hidden activation matrices (with
a handful of injected large-magnitude outlier columns), computes the
per-column absmax and threshold with a NumPy oracle, and checks that your
returned index set is exactly identical (same elements, any duplicate or
extra/missing index fails the trial).
