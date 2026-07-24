## Context

Quantization is a key technique for reducing the memory footprint and computational cost of neural networks.  
Two common 8‑bit quantization schemes are:

* **Linear Int8** – a symmetric scaling that maps the range $[-s,\, s]$ to the integer interval $[-128,\,127]$.  
  The scale factor is chosen as $s = \frac{\max |v|}{127}$.

* **Dynamic Map (Asymmetric)** – a per‑block mapping that uses the block’s minimum and maximum values.  
  It maps $[m,\, M]$ to the integer interval $[0,\,255]$, where $m=\min v$ and $M=\max v$.  
  The dequantized value is then reconstructed as $\hat{v}=q\cdot \frac{M-m}{255}+m$.

The quality of a quantization scheme can be measured by the mean‑squared error (MSE) between the original block $v$ and its reconstruction $\hat v$.  
For each input block we want to know which scheme yields the smaller MSE.

## Task

Implement the function

```python
def quant_winner_per_block(v_blocks: list[np.ndarray]) -> np.ndarray:
    ...
```

* `v_blocks` is a list of 2‑D NumPy arrays, each representing a second‑moment block.  
* The function must return a one‑dimensional integer array of the same length as `v_blocks`.  
  For every block it should contain `0` if Dynamic Map has lower or equal MSE, otherwise `1` (Linear Int8 wins).  
* All computations must use NumPy only; no explicit Python loops over elements.  
* The returned array must be of dtype `int32`.

## Example

```python
import numpy as np

blocks = [
    np.array([[0.5, -0.2], [1.3, 0.7]]),
    np.array([[-1.0, 2.0], [-0.5, 0.5]])
]

winners = quant_winner_per_block(blocks)
print(winners)   # e.g., array([0, 1])
```

## What the gate checks

The grader generates a handful of random blocks and computes the *oracle* winner for each block using NumPy.  
Your implementation must return exactly the same labels; otherwise the `exact_match` metric fails.
