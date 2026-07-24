## Context

Attention layers often store key and value tensors in reduced precision to save
memory bandwidth. FP8-style quantization requires a scale that maps floating
point values into a small integer range and a dequantization step before the
attention computation.

For a tensor $X$, symmetric quantization with one scale is:

$$s = \frac{\max(|X|)}{127}, \qquad \hat{X} = s \cdot \operatorname{clip}(\operatorname{round}(X/s), -127, 127).$$

A per-tensor scale uses one $s$ for all attention heads. A per-head scheme
computes a separate scale $s_h$ for each head. When one head contains unusually
large values, the shared scale can waste precision on other heads.

For query $Q$, key $K$, and value $V$, attention output is:

$$
\operatorname{Attn}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V .
$$

The quantization error can be measured by comparing the quantized output with
the original floating point attention output.

## Task

Implement `choose_kv_fp8_scheme(Q, K, V)`.

The inputs are NumPy arrays with shapes:

- `Q`: $(H, S, d)$ queries
- `K`: $(H, S, d)$ keys
- `V`: $(H, S, d)$ values

where $H$ is the number of attention heads, $S$ is sequence length, and $d$ is
the head dimension.

Return a tuple:

```python
(tensor_error, head_error, scheme)
```

where:

- `tensor_error` is the relative L2 error of attention output after quantizing
  both `K` and `V` using one shared scale per tensor.
- `head_error` is the relative L2 error after quantizing `K` and `V` with one
  scale per head.
- `scheme` is the string `"per_head"` if `head_error` is smaller, otherwise
  `"per_tensor"`.

Use NumPy operations only. The errors must be computed against the original
attention output.

## Example

```python
import numpy as np

Q = np.random.default_rng(0).normal(size=(2, 4, 8))
K = np.random.default_rng(1).normal(size=(2, 4, 8))
V = np.random.default_rng(2).normal(size=(2, 4, 8))

tensor_error, head_error, scheme = choose_kv_fp8_scheme(Q, K, V)
```

The returned scheme should identify which quantization granularity produces the
lower attention output error.

## What the gate checks

The gate uses a NumPy oracle that performs the attention computation and both
quantization schemes independently. It checks that the returned error values
match the oracle within numerical tolerance and that the selected scheme is the
oracle's lower-error choice. The fixtures include an outlier head where a
single shared scale loses precision compared with per-head scales.
