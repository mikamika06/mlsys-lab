## Context

NumPy broadcasting lets arrays with compatible shapes participate in elementwise
operations without manually expanding their storage. For an operation such as

$$
Y = A + B \cdot C ,
$$

each output element is computed independently. If the output index is
$i=(i_0,\dots,i_{k-1})$, broadcasting chooses the corresponding input index by
using the original dimension when it exists and using index $0$ for dimensions
that were expanded.

For two dimensions, an input with shape $(1,m)$ can be broadcast against an
output with shape $(n,m)$ because the first dimension is repeated. The element
formula is still

$$
Y_{ij}=A_{ij}+B_{ij}C_{ij}.
$$

A from-scratch implementation must perform the index mapping itself rather than
letting NumPy create the broadcasted operation.

## Task

Implement `broadcast_add_mul(A, B, C)`:

```python
def broadcast_add_mul(A, B, C):
    ...
```

The inputs are NumPy arrays with shapes that are valid for NumPy broadcasting.
Return a NumPy array containing the elementwise result of

$$
A + B \cdot C .
$$

Support the provided 2-D broadcast cases. The implementation must compute each
output element explicitly with Python index arithmetic and loops. Do not use
NumPy broadcasting for the operation itself.

## Example

```python
import numpy as np

A = np.array([[1.0, 2.0, 3.0]])
B = np.array([[2.0, 4.0, 6.0],
              [1.0, 3.0, 5.0]])
C = np.array([[10.0, 10.0, 10.0]])

Y = broadcast_add_mul(A, B, C)

# [[21. 42. 63.]
#  [11. 32. 53.]]
```

## What the gate checks

The numeric result is compared with a NumPy oracle using maximum absolute error:

$$
\max_i |Y_i-\hat{Y}_i|.
$$

The error must be at most $10^{-6}$.

A tracing check also records Python line events while the function runs. The
implementation must execute enough per-element Python work to demonstrate an
explicit elementwise loop rather than a single NumPy broadcasted expression.
