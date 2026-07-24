## Context

Quantizing a floating‑point tensor to an integer representation is common in deep learning inference. A **per‑tensor** scheme uses a single scale and zero point for the entire matrix, while a **per‑channel** scheme computes a distinct scale (and optionally zero point) for each column (or channel).

For an $n\times d$ weight matrix $W$, let $\text{scale}_t$ be the range of all entries divided by $255$ (the number of values in unsigned 8‑bit), and $\text{scale}_{c,i}$ the same but per column.  
The quantized value for entry $w_{ij}$ is

$$
q_{ij} = \operatorname{clip}\!\bigl(\,\bigl\lfloor\,\tfrac{w_{ij}-\min_j}{\text{scale}}\bigr\rceil,\,0,\,255\bigr),
$$

where $\min_j$ is the minimum of column $j$.  
Dequantization recovers a float approximation

$$
\hat w_{ij} = q_{ij}\,\text{scale}_{c,i} + \min_j .
$$

The **channel‑wise relative error** used by this task is

$$
\operatorname{rel}_i = \frac{\lVert W_i - \hat W_i\rVert_2}{\lVert W_i\rVert_2},
$$

averaged over all $d$ columns.  A lower value indicates that the per‑channel scheme preserves more of the original tensor’s structure.

## Task

Implement `compare_quantization(W)`:

```python
def compare_quantization(W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ...
```

* `W` is a 2‑D NumPy array of shape `(n, d)` with arbitrary real values.  
* The function must return two arrays of the same shape as `W`:  
  * **per‑tensor approximation** – quantize all entries using a single scale and dequantize back to float.  
  * **per‑channel approximation** – quantize each column independently (using its own min and max) and dequantize back to float.  
* Use only NumPy operations; no explicit Python loops over elements or columns.  
* The returned arrays must be of type `float64`.

## Example

```python
import numpy as np
W = np.array([[0, -1], [2, 3]])
per_tensor, per_channel = compare_quantization(W)
print(per_tensor)
# [[0.         0.        ]
#  [2.00000000 3.00000000]]
print(per_channel)
# [[0.          -1.        ]
#  [2.00000000 3.00000000]]
```

## What the gate checks

The grader computes the **channel‑wise relative error** between `W` and the per‑channel approximation returned by your function.  
The solution must satisfy

$$
\operatorname{rel}_{\text{chan}} \;\leq\; 0.05 .
$$

A fully vectorised implementation that correctly handles zero‑range columns will pass this gate. A broken or overly simplistic implementation will produce a higher error and fail.
