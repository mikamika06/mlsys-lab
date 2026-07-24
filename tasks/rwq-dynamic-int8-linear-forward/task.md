## Context

"Dynamic" int8 quantization for a linear layer quantizes the **weight**
once (ahead of time, per output channel) and quantizes the **activation**
on the fly for every forward call, from that call's own min/max — this is
exactly the scheme PyTorch's `torch.quantization` dynamic-quant path and
many inference runtimes use for `Linear` layers.

**Weight** $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ is
quantized **per output channel** (per row), symmetric, to signed int8 in
$[-127, 127]$:

$$
s^{W}_o = \frac{\max_j |W_{o,j}|}{127}, \qquad
q^{W}_{o,j} = \mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{W_{o,j}}{s^{W}_o}\right),\, -127,\, 127\right)
$$

**Activation** $X \in \mathbb{R}^{b \times d_{\text{in}}}$ is quantized
**per tensor**, asymmetric, to unsigned uint8 in $[0, 255]$, using that
call's own dynamic range:

$$
s^{X} = \frac{\max(X) - \min(X)}{255}, \qquad
z^{X} = \mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{-\min(X)}{s^{X}}\right),\, 0,\, 255\right)
$$
$$
q^{X}_{b,j} = \mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{X_{b,j}}{s^{X}} + z^{X}\right),\, 0,\, 255\right)
$$

The forward pass runs the matmul in the **integer** domain (subtracting the
zero-point first, exactly as real int8 kernels do) and dequantizes with both
scales at the end:

$$
Y_{b,o} = s^{X} \, s^{W}_o \sum_{j} \left(q^{X}_{b,j} - z^{X}\right) q^{W}_{o,j}
$$

which approximates the unquantized $Y = X W^\top$.

## Task

Implement `int8_linear_forward`:

```python
def int8_linear_forward(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    ...
```

* `X` — activations, shape $(b, d_{\text{in}})$.
* `W` — weights, shape $(d_{\text{out}}, d_{\text{in}})$.

Quantize `W` per-channel symmetric int8 and `X` per-tensor asymmetric
uint8 exactly as above, run the integer matmul with the zero-point
correction, dequantize, and return `Y`, shape $(b, d_{\text{out}})$.

If a channel of `W` is entirely zero (`max(|W_o|) == 0`), treat its scale
as `1.0` instead of dividing by zero. If `X` is constant (`max(X) ==
min(X)`), treat the range as a tiny positive number instead of zero.

## Example

```python
import numpy as np
X = np.array([[1.0, -2.0, 3.0]])
W = np.array([[1.0, 0.0, -1.0], [0.5, 0.5, 0.5]])
Y = int8_linear_forward(X, W)
# Y approximates X @ W.T == [[1*1 + (-2)*0 + 3*(-1), 1*0.5 + -2*0.5 + 3*0.5]]
#                          == [[-2.0, 1.0]]
```

## What the gate checks

A single **rel_err** gate builds several random `(X, W)` pairs, runs the
same quantize -> int matmul -> dequantize pipeline described above as a
NumPy oracle, and compares your output to it with the global relative L2
error, requiring `< 1e-3`. Any shape mismatch or exception fails the gate.
