## Context

FP8 key/value (KV) caches reduce memory bandwidth by storing tensors in an 8-bit floating-point format. The e4m3 format has a limited dynamic range, so values larger than the representable maximum are saturated.

For a tensor $x$, a production quantization path usually computes an absolute maximum value

$$a = \max_i |x_i|$$

and chooses a scale

$$s = \frac{a}{448},$$

where $448$ is the largest magnitude representable by the e4m3 FP8 format. Values are quantized as

$$q_i = \mathrm{fp8}\left(\frac{x_i}{s}\right),$$

then reconstructed as

$$\hat{x}_i = s \cdot \mathrm{dequant}(q_i).$$

Using $s=1$ for tensors with large values causes saturation because values above $448$ are clipped before being stored. This can significantly change attention results.

Scaled dot-product attention is computed as

$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V.$$

The KV cache must be dequantized with the same calibrated scale before it is used.

## Task

Implement `fp8_attention_output(Q, K, V)`.

The function receives three `float64` list:

- `Q` with shape $(m,d)$
- `K` with shape $(n,d)$
- `V` with shape $(n,d)$

It must simulate an FP8 e4m3 KV cache correctly:

1. Compute separate scales for `K` and `V` using the maximum absolute value divided by $448$.
2. Quantize and dequantize `K` and `V` using an e4m3-like representation. The implementation should clamp values outside the FP8 range before storing.
3. Compute the attention output using the reconstructed tensors.

Return a `float64` array of shape $(m,d)$.

Do not use a fixed scale of $1.0`; the purpose of the task is to avoid saturation caused by missing amax calibration.

## Example

```python

Q = [[1.0, 0.5]]
K = [[500.0, 0.0], [-500.0, 1.0]]
V = [[2.0, 3.0], [4.0, 5.0]]

out = fp8_attention_output(Q, K, V)
```

The output is computed after the KV tensors have gone through the calibrated FP8 round trip.

## What the gate checks

The gate computes an independent Python oracle that performs e4m3 quantization with amax-based scaling and compares the submitted attention output.

The returned `max_abs_err` must satisfy

$$\max_i |\mathrm{output}_i-\mathrm{oracle}_i| \le 0.01.$$

A path that keeps the scale fixed at $1.0$ will saturate large KV values and fail the error threshold.
