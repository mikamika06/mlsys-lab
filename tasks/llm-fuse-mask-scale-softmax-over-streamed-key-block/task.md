## Context

In the context of neural networks, particularly in transformer architectures, the scaled dot-product attention mechanism is crucial for processing sequences. The attention mechanism computes a weighted sum of values based on the similarity of keys and queries. The scaled dot-product attention can be expressed mathematically as:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

where $M$ is a mask that prevents attending to certain positions (e.g., future tokens in causal attention).

## Task

Implement the function `fuse_mask_scale_softmax(keys, values, mask, scale)`:

```python
def fuse_mask_scale_softmax(keys: np.ndarray, values: np.ndarray, mask: np.ndarray, scale: float) -> np.ndarray:
    ...
```

This function takes:
- `keys`: a 2-D NumPy array of shape $(n, d_k)$ representing the keys.
- `values`: a 2-D NumPy array of shape $(n, d_v)$ representing the values.
- `mask`: a 2-D NumPy array of shape $(n, n)$ for the causal mask.
- `scale`: a float representing the scaling factor.

The function should return a 2-D NumPy array of shape $(n, n)$ containing the attention scores after applying the mask and softmax. The result must be of type `float64`.

## Example

```python
import numpy as np

keys = np.random.rand(10, 64)
values = np.random.rand(10, 64)
mask = np.random.rand(10, 10) * -1e9  # Causal mask
scale = 1 / np.sqrt(64)

attention_scores = fuse_mask_scale_softmax(keys, values, mask, scale)
```

## What the gate checks

The gate checks the maximum absolute error between the output of your implementation and a reference implementation. The maximum absolute error must be less than $1 \times 10^{-5}$. This ensures that your implementation is both accurate and efficient, as it must compute the attention scores in a single pass without materializing the full score matrix.
