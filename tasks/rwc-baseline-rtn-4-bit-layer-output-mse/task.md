## Context

Quantization is a key technique for reducing the memory and compute footprint of neural networks.  
Round‑to‑nearest (RTN) quantization maps each real weight $w$ to an integer $q$ by  

$$ q = \operatorname{clip}\!\bigl(\,\operatorname{round}(w / s),\, -8, 7\bigr) , $$

where the scale $s>0$ is chosen per group of weights.  
For a signed 4‑bit representation the integer range is $[-8,7]$.  
After quantization we recover an approximate real weight by  

$$ \hat w = q\, s . $$  

The quality of a quantized layer can be measured by the mean‑squared error (MSE) between the outputs of the original and quantized weight matrices when applied to a set of calibration activations $X$:

$$
\operatorname{MSE} = \frac{1}{n}\sum_{i=1}^{n}
   \bigl\| W\,x_i - \hat W\,x_i \bigr\|^2 .
$$

Here $W$ is the real weight matrix, $\hat W$ its quantized counterpart, and $X$ contains $n$ column‑vectors $x_i$.

## Task

Implement a function that computes this MSE for a given weight matrix $W$ and calibration activations $X$. The function must perform **per‑group** RTN 4‑bit quantization with a fixed group size of 16 columns. Padding should be applied so that each row is divided into an integer number of groups; padded entries are treated as zeros.

```python
def quantize_layer_output_mse(W: np.ndarray,
                              X: np.ndarray,
                              group_size: int = 16) -> float:
    """
    Return the MSE between W @ X and the RTN‑4bit quantized version of W applied to X.
    """
```

The function must use only NumPy operations (no explicit Python loops).  
Return a `float` with type `np.float64`.

## Example

```python
import numpy as np
W = np.array([[0.5, -1.2], [3.4, 0.0]])
X = np.array([[1.0], [-1.0]])
mse_val = quantize_layer_output_mse(W, X)
print(mse_val)   # e.g. 0.123456789
```

## What the gate checks

The grader computes an oracle MSE using the same algorithm and compares it to the value returned by your function.  
Your implementation must produce a difference of at most $10^{-6}$ from the oracle. Any deviation larger than this threshold will cause the task to fail.
