## Context

Consider a chain of $K$ elementwise operations applied to an array of $N$ elements. Each operation has the form $y = f(x)$, where $x$ and $y$ are arrays of $N$ elements. When these operations are executed one after another in a loop, we have two strategies: unfused and fused. The unfused strategy stores the output of each operation in a separate array, while the fused strategy combines the operations into a single loop.

In terms of memory traffic, the unfused strategy requires $K+1$ full-tensor read/write operations, resulting in a total of $2N(K+1)$ bytes transferred between memory and the processing unit. On the other hand, the fused strategy requires only two memory accesses: one read and one write, resulting in a total of $2N$ bytes transferred.

The memory traffic reduction from fusion can be quantified as follows:
$$\text{reduction} = \frac{2N(K+1) - 2N}{2N(K+1)} = \frac{K-1}{K+1}$$
This reduction can be significant for large $K$ and $N$.

## Task

Implement two functions: `dram_traffic_fused(N, K, dtype_size)` and `dram_traffic_unfused(N, K, dtype_size)`, which calculate the DRAM traffic in bytes for the fused and unfused strategies, respectively.

```python
import numpy as np

def dram_traffic_fused(N, K, dtype_size):
    # Calculate DRAM traffic for fused strategy
    pass

def dram_traffic_unfused(N, K, dtype_size):
    # Calculate DRAM traffic for unfused strategy
    pass
```

## Example

```python
N = 1000
K = 10
dtype_size = 8  # bytes

fused_traffic = dram_traffic_fused(N, K, dtype_size)
unfused_traffic = dram_traffic_unfused(N, K, dtype_size)

print(f"Fused traffic: {fused_traffic} bytes")
print(f"Unfused traffic: {unfused_traffic} bytes")
```

## What the gate checks

The gate checks if the calculated DRAM traffic for both strategies matches the expected values computed using the NumPy oracle.
