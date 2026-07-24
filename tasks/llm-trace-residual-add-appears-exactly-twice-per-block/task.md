## Context

In a transformer block the input tensor $x \in \mathbb{R}^{n\times d}$ is first processed by self‑attention, then added back to the original input (the *residual*).  
The result of that addition is fed into a feed‑forward network and **again** added back to its own output.  Thus each block contains exactly two residual additions.

Mathematically:

$$
\begin{aligned}
a &= x W_1 + b_1,\\[4pt]
y &= x + a,\\[4pt]
b &= y W_2 + b_2,\\[4pt]
z &= y + b.
\end{aligned}
$$

The final output $z$ is returned.

## Task

Implement the function `transformer_block` that performs the computation above **using a helper** called `add_residual(a,b)` for each residual addition.  The helper must be invoked exactly twice per call to `transformer_block`.

```python
def add_residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ...

def transformer_block(x: np.ndarray,
                      w1: np.ndarray, b1: np.ndarray,
                      w2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    ...
```

All operations should be vectorised with NumPy; no Python loops are allowed.  The function must return a `float64` array of shape `(n,d)`.

## Example

```python
import numpy as np
x = np.array([[1., 2.], [3., 4.]])
w1 = np.eye(2)
b1 = np.zeros(2)
w2 = np.ones((2, 2))
b2 = np.zeros(2)

z = transformer_block(x, w1, b1, w2, b2)
print(z)
# [[ 3.  5.]
#  [ 7.  9.]]
```

## What the gate checks

Two metrics are evaluated:

* **`residual_add_count`** – a trace counter that records how many times `add_residual` is called during a single call to `transformer_block`.  The value must be exactly `2`.

* **`rel_err`** – the global relative L2 error between your output and a reference implementation.  It must not exceed $10^{-9}$.

The trace counter uses Python’s `sys.settrace`; therefore any deviation from two calls (e.g., using plain addition or an extra helper call) will cause the gate to fail.  The numerical check ensures that the arithmetic is correct even if the helper is used correctly.
