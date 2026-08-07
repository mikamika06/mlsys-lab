## Context

In many transformer‑style models the key tensor $K \in \mathbb{R}^{N\times D}$ is quantized before it is used in a dot‑product attention operation.  
Quantization can be performed along either the **row** axis (quantizing each token independently) or the **column** axis (quantizing each feature channel independently).  
A common heuristic for choosing the better axis is to look at the *dynamic range* of each group:

$$\operatorname{range}(G)=\max(G)-\min(G).$$

For a fixed axis we compute this range for every group along that axis and then take the variance of those ranges.  The axis with the larger variance tends to give a lower reconstruction error when the groups are quantized separately, because it captures more of the spread in the data.

The variance of a set $\{r_i\}$ is

$$\operatorname{var}(\{r_i\})=\frac1n\sum_{i=1}^{n}(r_i-\bar r)^2,$$

where $\bar r$ is the mean range. In Python this can be expressed with `max` and `min` (for peak‑to‑peak) and `statistics.variance`.

## Task

Implement a function

```python
def classify_quant_axis(K: list[list[float]]) -> int:
    ...
```

that takes a 2‑D list $K$ of shape $(N,D)$ and returns the integer `0` if quantizing along rows is preferable, or `1` if quantizing along columns is preferable.  
The decision should be based on comparing

$$\operatorname{var}\bigl(\{\operatorname{range}(K_i)\}_{i=0}^{N-1}\bigr) \quad\text{vs}\quad
\operatorname{var}\bigl(\{\operatorname{range}(K^j)\}_{j=0}^{D-1}\bigr),$$

where $K_i$ denotes the $i$‑th row and $K^j$ the $j$‑th column.  
Use only Python operations; no explicit Python loops.

## Example

```python
K = [[0, 1, 2],
              [3, 4, 5]]
# Row ranges: [2, 2] → variance 0
# Column ranges: [3, 3, 3] → variance 0
# Ties are broken by choosing axis 0.
print(classify_quant_axis(K))  # 0
```

## What the gate checks

The grader computes an *oracle* that applies exactly the same logic to a set of test tensors and verifies that your function returns the identical integer for each case.  No other metrics are evaluated.
