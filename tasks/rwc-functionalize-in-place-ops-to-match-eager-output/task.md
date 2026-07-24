## Context

In many numerical libraries, operations such as addition, rectified linear unit (ReLU) and copying are implemented in‑place to save memory.  
The *in‑place* variant of an operation mutates its input tensors; the *out‑of‑place* or *functional* variant returns a new tensor while leaving all inputs untouched.

For example, the ReLU function is defined as

$$\operatorname{ReLU}(x) = \max(0,x).$$

When implemented in‑place it would modify `x`; when functional it must produce a fresh array with the same shape and values but without altering `x`.

## Task

Implement three pure functions that perform the same computations as their in‑place counterparts but **do not mutate any input arguments**:

```python
def functional_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return a new array equal to a + b."""

def functional_relu(x: np.ndarray) -> np.ndarray:
    """Return a new array with ReLU applied elementwise."""

def functional_copy(a: np.ndarray) -> np.ndarray:
    """Return a copy of the input array."""
```

All functions must work for arbitrary NumPy arrays of any shape and dtype.  
The returned arrays should be independent copies; modifying them later must not affect the original inputs.

## Example

```python
import numpy as np
a = np.array([[1, -2], [3, 0]])
b = np.array([[4, 5], [-6, 7]])

c_add = functional_add(a, b)
# [[5, 3],
#  [-3, 7]]

c_relu = functional_relu(a)
# [[1, 0],
#  [3, 0]]

c_copy = functional_copy(b)
# same as b but a distinct array
```

After calling the functions, `a` and `b` remain unchanged.

## What the gate checks

Two metrics are evaluated:

* **max_abs_err_output** – the maximum absolute difference between each function’s output and the NumPy reference (`a + b`, `np.maximum(0,x)`, `a.copy()`). It must be ≤ $10^{-12}$.
* **max_abs_err_input** – the maximum absolute difference between any input array before and after the call. Since inputs should not change, this value must be 0.

The grader runs each function on randomly generated tensors and compares the results numerically while also verifying that all original inputs are byte‑exactly unchanged. A correct implementation passes both gates; a broken one (e.g., mutating an input) fails at least one gate.
