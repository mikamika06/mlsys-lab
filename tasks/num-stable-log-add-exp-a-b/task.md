## Context

Log-domain arithmetic is used when adding values that may be too large or too
small to represent directly. The log-sum-exp operation for two inputs is

$$
\operatorname{logaddexp}(a,b) = \log(\exp(a) + \exp(b)).
$$

A direct implementation can overflow when $a$ or $b$ is large. For example,
$\exp(1000)$ cannot be represented in standard floating point arithmetic.

A stable rearrangement factors out the larger input:

$$
\log(\exp(a)+\exp(b))
=
\max(a,b) + \log(1+\exp(-|a-b|)).
$$

The second term is computed with `log1p`, which preserves precision when its
argument is close to zero.

## Task

Implement `stable_log_add_exp(a, b)`.

The function receives two real-valued NumPy arrays or array-like values and must
return a NumPy array containing

$$
\max(a,b) + \log(1+\exp(-|a-b|)).
$$

Use NumPy operations only. The result must be `float64` and must match NumPy's
`np.logaddexp` behavior for finite inputs.

The function signature is:

```python
def stable_log_add_exp(a, b):
    ...
```

## Example

```python
import numpy as np

a = np.array([1000.0, 1.0])
b = np.array([999.0, 2.0])

out = stable_log_add_exp(a, b)
# approximately:
# [1000.31326169, 2.31326169]
```

## What the gate checks

The gate computes a reference result using NumPy's `np.logaddexp` on several
arrays containing normal and large-magnitude values. The submitted function is
graded using the relative error

$$
\mathrm{rel\_err}
=
\frac{\lVert y-\hat{y}\rVert_2}{\lVert y\rVert_2+10^{-12}},
$$

where $y$ is the NumPy reference and $\hat{y}$ is the submitted result.

The relative error must satisfy $\mathrm{rel\_err}<10^{-13}$. A direct
implementation using $\log(\exp(a)+\exp(b))$ can overflow and will not pass.
