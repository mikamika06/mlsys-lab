## Context

Inference systems often quantize weights or KV-cache tensors to reduce memory usage. A simple approach uses one scale for the entire tensor. This is called per-tensor quantization.

For a tensor $W$, symmetric int8 quantization uses

$$
q = \operatorname{clip}\left(\operatorname{round}\left(\frac{W}{s}\right), -127, 127\right),
$$

where $s$ is the scale and the dequantized tensor is

$$
\hat{W} = q s .
$$

Per-tensor quantization uses one scale:

$$
s = \frac{\max(|W|)}{127}.
$$

If different channels have different magnitudes, large channels determine the scale and smaller channels lose precision. Per-channel quantization computes one scale for each channel. For row $i$,

$$
s_i = \frac{\max_j(|W_{ij}|)}{127}.
$$

The reconstruction error is measured with mean per-channel relative error:

$$
\operatorname{channel\_rel\_err}(W,\hat{W}) =
\frac{1}{n}\sum_i
\frac{\lVert \hat{W}_i-W_i\rVert_2}
{\lVert W_i\rVert_2 + 10^{-12}} .
$$

## Task

Implement `compare_quant_errors(W)`:

```python
def compare_quant_errors(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ...
```

The input is a 2-D floating point NumPy array. Rows represent independent channels.

Return two reconstructed tensors:

1. The first output must be `W` reconstructed with per-tensor symmetric int8 quantization.
2. The second output must be `W` reconstructed with per-channel symmetric int8 quantization, using one scale per row.

Use NumPy operations and do not modify the input array.

## Example

```python
import numpy as np

W = np.array([
    [0.1, 0.2, 0.3],
    [8.0, -7.0, 6.0],
], dtype=np.float64)

tensor_out, channel_out = compare_quant_errors(W)

# tensor_out uses one scale based on the largest value in W
# channel_out uses a separate scale for each row
```

## What the gate checks

The grader computes the quantization reference directly from the equations above using NumPy.

It checks that the returned per-channel reconstruction matches the oracle, that its mean per-channel relative error is small, and that it improves over per-tensor quantization on tensors where channel magnitudes differ.

The measured error is

$$
\operatorname{channel\_rel\_err}(W,\hat{W})
$$

computed independently by the grader.
