## Context

Modern hardware accelerators (GPUs) provide specialized compute paths for
different types of weight sparsity.  The three common serving paths are:

- **Dense**: standard general matrix multiply (GEMM); best for tensors with
  density close to 1 (few zeros).
- **CSR (Compressed Sparse Row)**: a sparse format that stores only non-zero
  values and their column indices.  It becomes efficient when the fraction of
  zeros, the *sparsity ratio*

  $$ s = \frac{\#\text{zeros}}{\text{total elements}} $$

  is high (typically $s \ge 0.5$) and the non-zero pattern is unstructured.
- **Tensor-Core 2:4**: a structured sparsity pattern supported by NVIDIA tensor
  cores.  Each row of the weight matrix is partitioned into groups of 4
  consecutive columns, and every group must contain **exactly 2 non‑zero**
  entries.  This constraint enables the hardware to skip zero loads
  automatically.  A matrix that satisfies this condition is said to have *valid
  2:4 structured sparsity*.

Choosing the right path for a given weight tensor is a critical first step in
many ML deployment pipelines.

## Task

Implement the function `route_tensor(weights)` that inspects the input weight
tensor and returns the appropriate serving path.

```python
def route_tensor(weights: np.ndarray) -> str:
    """
    Determine the recommended serving path for a weight tensor.

    Parameters
    ----------
    weights : np.ndarray
        A 2‑D NumPy array of shape (out_features, in_features).

    Returns
    -------
    str
        One of "tensor-core", "csr", or "dense", following the rules:
        1. If the tensor has valid 2:4 structured sparsity (each row’s columns
           are divisible into groups of 4, each group has exactly 2 non‑zeros),
           return "tensor-core".
        2. Otherwise, if the sparsity ratio $s \ge 0.5$, return "csr".
        3. Otherwise, return "dense".
    """
```

The algorithm must operate **only** with NumPy functions on the input array.
No Python loops over rows or columns are allowed — use vectorised operations.

## Example

```python
import numpy as np

# Dense weights -> "dense"
w_dense = np.ones((4, 4))
print(route_tensor(w_dense))   # "dense"

# 2:4 structured weights -> "tensor-core"
# Generate a matrix with exactly two non‑zeros per block of four columns
rng = np.random.default_rng(0)
w_2_4 = np.zeros((2, 8))
for start in range(0, 8, 4):
    cols = rng.choice(4, 2, replace=False)   # positions within block
    w_2_4[:, start + cols] = rng.uniform(-1, 1, (2, 2))
print(route_tensor(w_2_4))   # "tensor-core"

# Highly sparse unstructured weights (70% zeros) -> "csr"
w_sparse = rng.uniform(-1, 1, (4, 16))
w_sparse[w_sparse < 0.7] = 0.0          # keep ~30% non‑zeros
print(route_tensor(w_sparse))           # "csr"
```

## What the gate checks

The gate runs `route_tensor` on a collection of weight matrices covering dense,
moderately sparse, highly sparse (≥50%), and valid 2:4 structured cases.  For
every input the returned string must **exactly** match the oracle result.  A
single mismatch yields score 0.0, so the solution must correctly implement all
three branches.

No additional libraries beyond NumPy are allowed.  The solution must not use
Python iteration over rows or columns.
