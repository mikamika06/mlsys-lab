## Context

In many deep‑learning libraries a weight matrix of shape \((N, H)\) is stored in mixed precision to save memory.  
A common strategy compresses the majority of columns to **int8** (1 byte each) while keeping a small set of *outlier* columns at **fp16** (2 bytes).  

Let \(H\) be the total number of columns and let \(k\) denote the number of outlier columns that remain fp16.  
The effective memory required per row‑group is

$$
B_{\text{eff}} = (H-k)\cdot 1 + k \cdot 2 .
$$

A fully fp16 implementation would use

$$
B_{\text{fp16}} = 2\,H
$$

bytes.  
The *size ratio* relative to the fp16 baseline is therefore

$$
R = \frac{B_{\text{eff}}}{B_{\text{fp16}}}
   = \frac{(H-k)\cdot 1 + k \cdot 2}{2\,H}.
$$

Accurate computation of \(B_{\text{eff}}\) and \(R\) is essential for memory budgeting in production systems.

## Task

Implement the function `mixed_precision_memory_accounting` that takes two integers, `H` (total columns) and `k` (outlier columns), and returns a tuple `(bytes_per_row_group, size_ratio)` as described above. The result must be of type `float`.

```python
def mixed_precision_memory_accounting(H: int, k: int) -> tuple[float, float]:
    """
    Compute effective bytes per row-group and the ratio to a fully fp16 implementation.

    Parameters
    ----------
    H : int
        Total number of columns (hidden dimension).
    k : int
        Number of outlier columns kept at 2‑byte precision.
        The remaining H-k columns are compressed to 1 byte each.

    Returns
    -------
    bytes_per_row_group : float
        Bytes required per row-group: (H - k) * 1 + k * 2.
    size_ratio : float
        Ratio of the computed bytes to the baseline fp16 memory usage:
        ((H - k) * 1 + k * 2) / (2 * H).
    """
    ...
```

## Example

```python
>>> mixed_precision_memory_accounting(1024, 0)
(1024.0, 0.5)
>>> mixed_precision_memory_accounting(2048, 512)
(3072.0, 0.75)
```

## What the gate checks

The grader computes a reference implementation using NumPy and compares the returned size ratio to that reference.  
It reports the **relative error**

$$
\mathrm{rel\_err} = \frac{|R_{\text{candidate}} - R_{\text{reference}}|}
                         {R_{\text{reference}} + 10^{-12}}
$$

The solution must satisfy $\mathrm{rel\_err}\le 10^{-9}$ on a set of test cases.  
Any deviation beyond this threshold causes the gate to fail.
