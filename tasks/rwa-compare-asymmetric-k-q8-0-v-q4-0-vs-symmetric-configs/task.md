## Context

Transformer attention computes output vectors from query, key, and value matrices. For one query $q$, keys $K$, and values $V$, the attention output is

$$
\mathrm{Attn}(q,K,V)=\mathrm{softmax}\left(\frac{qK^\top}{\sqrt{d}}\right)V .
$$

A KV cache stores $K$ and $V$ tensors between decoding steps. Quantizing the cache reduces memory usage, but different tensor types can have different accuracy costs.

This task models three configurations:

- symmetric $q8/q8$: both keys and values use 8-bit symmetric quantization.
- symmetric $q4/q4$: both keys and values use 4-bit symmetric quantization.
- asymmetric precision $q8/q4$: keys use 8-bit quantization while values use 4-bit quantization.

For a tensor block $x$, symmetric quantization uses

$$
s=\frac{\max(|x|)}{2^{b-1}-1}, \qquad \hat{x}=\mathrm{round}(x/s)s .
$$

The attention result from quantized tensors can be compared with the full precision result using maximum absolute error.

The asymmetric configuration keeps the key vectors more accurate. Since keys determine attention scores, this can preserve attention quality better than reducing both keys and values to 4-bit precision at a similar cache size.

## Task

Implement `kv_config_attention_errors(K, V, q)`:

```python
def kv_config_attention_errors(K: np.ndarray, V: np.ndarray, q: np.ndarray) -> np.ndarray:
    ...
```

Inputs:

- `K` is a float array with shape $(n,d)$ containing key vectors.
- `V` is a float array with shape $(n,d)$ containing value vectors.
- `q` is a float array with shape $(d,)$ containing one query vector.

Return a float64 array of length 3 with errors in this order:

1. $q8/q8$
2. $q4/q4$
3. $q8/q4$

Each value is the maximum absolute difference between the full precision attention output and the output using the quantized $K,V$ pair.

Implement the quantization and attention computation directly with NumPy.

## Example

```python
import numpy as np

K = np.array([[1.0, 0.5], [0.0, 2.0], [-1.0, 1.0]])
V = np.array([[1.0, 0.0], [0.5, 1.0], [-1.0, 0.5]])
q = np.array([0.5, 1.0])

errors = kv_config_attention_errors(K, V, q)
# errors[0] is the q8/q8 error
# errors[1] is the q4/q4 error
# errors[2] is the q8/q4 error
```

## What the gate checks

The gate computes the same quantization and attention calculation using an independent NumPy oracle. The returned three errors must match the oracle values with maximum absolute error at most $10^{-9}$.

The oracle also verifies the intended comparison property on generated cases: the higher-precision-key configuration $q8/q4$ should improve over the symmetric $q4/q4$ configuration while using a similar KV memory ratio.
