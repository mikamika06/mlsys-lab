## Context

In a transformer architecture the *residual stream* is added to the output of each block. If we denote by $x$ the input residual and by $\mathcal{B}$ the transformation performed inside one block (layer‑norm + attention + MLP), then after applying $n$ identical blocks we obtain
$$
x_n = \mathcal{B}^n(x) .
$$
The growth of the norm of the residual is an indicator of how much the network amplifies or dampens signals.  For a fixed block $\mathcal{B}$ we can quantify this by the ratio
$$
r(\mathcal{B}) = \frac{\lVert x_n\rVert_2}{\lVert x\rVert_2},
$$
where $\lVert\cdot\rVert_2$ is the Euclidean norm over all elements of the tensor.  In practice we evaluate this ratio for a random input and a fixed number of blocks.

## Task

Implement `residual_norm_growth(block_fn, x)`:

```python
def residual_norm_growth(block_fn: Callable[[np.ndarray], np.ndarray],
                         x: np.ndarray) -> float:
    ...
```

`block_fn` is a callable that accepts a NumPy array and returns an array of the same shape.  The function should apply `block_fn` **twelve** times in succession to the input tensor `x`, compute the Euclidean norm of the final result, divide by the norm of the original input, and return this ratio as a Python float.

The implementation must use only NumPy operations; no explicit Python loops over elements are allowed.  The function should work for any shape of `x` that is compatible with `block_fn`.

## Example

```python
import numpy as np

# A simple linear block: y = x @ W + b
rng = np.random.default_rng(0)
W = rng.standard_normal((64, 64))
b = rng.standard_normal(64)

def linear_block(x):
    return x @ W + b

x = rng.standard_normal((32, 64))

ratio = residual_norm_growth(linear_block, x)
print(ratio)   # e.g. 1.2345
```

## What the gate checks

The grader generates a random input tensor and a fixed linear block with known weights.  
It computes the reference ratio by applying the block twelve times using NumPy directly.  
Your implementation is called with the same arguments; the returned value must have a relative error
$$
\frac{|\,\text{your}\, - \,\text{reference}\,|}{|\text{reference}|}
\le 10^{-3}.
$$

If the ratio deviates by more than $0.1\%$ the solution fails.
