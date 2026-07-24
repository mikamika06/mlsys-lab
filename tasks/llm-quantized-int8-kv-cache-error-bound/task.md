## Context

In an autoregressive transformer, the *key/value cache* stores the activations of every token that has already been processed.  
During decoding a new query vector $q$ is compared to all cached keys $k_i$ via the dot‑product attention score

$$s_i = \frac{q^\top k_i}{\sqrt{d_k}},$$

where $d_k$ is the key dimension.  The softmax over $\{s_i\}$ yields a probability distribution that determines how much each cached value contributes to the output.

Quantizing the cache to 8‑bit integers reduces memory usage and can accelerate inference on hardware with fast integer arithmetic.  A common strategy is *per‑row symmetric scaling*: for each row $k_i$ we compute a scale factor $\alpha_i$ such that

$$\alpha_i = \frac{\max_j |k_{ij}|}{127},$$

and store the quantized values

$$\tilde{k}_{ij} = \operatorname{round}\!\left(\frac{k_{ij}}{\alpha_i}\right), \qquad
\tilde{k}_{ij}\in[-127,127].$$

The reconstructed float is $\hat{k}_i = \alpha_i\,\tilde{k}_i$.

Because the attention distribution depends on the dot products $q^\top k_i$, quantization introduces error.  We measure this error with the **mean Kullback–Leibler divergence** between the reference softmax (computed from full‑precision keys) and the softmax obtained after reconstructing the quantized cache.

## Task

Implement a function that takes the original key and value tensors in `float16` and returns an 8‑bit integer representation of the keys together with their per‑row scale factors:

```python
def kv_cache_quantize(keys_fp16: np.ndarray, values_fp16: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantizes the key tensor to int8 using a per-row symmetric scale.
    
    Parameters
    ----------
    keys_fp16 : np.ndarray of shape (n, d)
        The original key matrix in float16.
    values_fp16 : np.ndarray of shape (n, d)
        The value matrix in float16.  It is passed for API compatibility but
        not used by the quantization routine.

    Returns
    -------
    keys_int8 : np.ndarray of shape (n, d), dtype=int8
        Quantized key values.
    scales   : np.ndarray of shape (n,), dtype=float32
        The per‑row scale factors used for reconstruction.
    """
```

The function must be fully vectorised: no explicit Python loops over rows or columns.  It should clip the quantised values to the range $[-127,127]$ and use `float32` for the scales.

## Example

```python
import numpy as np

keys_fp16 = np.array([[0.1, -0.2], [0.3, 0.4]], dtype=np.float16)
values_fp16 = np.zeros_like(keys_fp16)          # dummy values

keys_int8, scales = kv_cache_quantize(keys_fp16, values_fp16)

print(keys_int8)
# [[  1  -2]
#  [ 24  32]]
print(scales)
# [0.00078125 0.003125 ]
```

Reconstructing the keys with `keys_int8.astype(np.float32) * scales[:, None]` yields values close to the original.

## What the gate checks

The grader computes, for several random test cases, the mean KL divergence between:

* the reference attention distribution obtained from the full‑precision keys,
* and the distribution obtained after reconstructing the quantised cache.

The metric `mean_kl` must satisfy

$$\text{mean\_kl} \le 0.03.$$

A correctly implemented per‑row symmetric scaling typically achieves a KL below $2\times10^{-2}$ on random data, so the threshold is generous yet nontrivial.
