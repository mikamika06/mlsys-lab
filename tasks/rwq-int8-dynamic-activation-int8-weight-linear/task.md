## Context

Production inference engines (TensorRT, GGML, PyTorch dynamic quantization) replace
the floating-point matrix multiply inside `nn.Linear` with lower-precision integer
arithmetic. A widely-used scheme quantizes **weights per output channel** and
**activations per token** at runtime, because activation ranges differ across
tokens while weight ranges are fixed after training.

**Symmetric per-channel weight quantization.** For row $i$ of $W \in \mathbb{R}^{N \times K}$:

$$
s_i^{w} = \frac{\max_{k}|w_{ik}|}{127}, \qquad
w_i^{\,\mathrm{int8}} = \mathrm{clip}\!\Bigl(\mathrm{round}\!\bigl(w_i / s_i^{w}\bigr),\; -128,\; 127\Bigr)
$$

**Symmetric per-token activation quantization.** For row $j$ of $X \in \mathbb{R}^{B \times K}$:

$$
s_j^{x} = \frac{\max_{k}|x_{jk}|}{127}, \qquad
x_j^{\,\mathrm{int8}} = \mathrm{clip}\!\Bigl(\mathrm{round}\!\bigl(x_j / s_j^{x}\bigr),\; -128,\; 127\Bigr)
$$

**Dequantized output.** The integer matmul accumulates in `int32`, then each element
is scaled back:

$$
Y_{ji} = s_j^{x}\, s_i^{w} \sum_{k=1}^{K} x_j^{\,\mathrm{int8}}[k] \;\cdot\; w_i^{\,\mathrm{int8}}[k]
$$

In matrix form:

$$
Y = \bigl(X^{\mathrm{int8}}\,(W^{\mathrm{int8}})^{\!\top}\bigr) \;\odot\; \bigl(s^{x}\,(s^{w})^{\!\top}\bigr)
$$

where $\odot$ denotes element-wise (Hadamard) multiplication and $s^x \in \mathbb{R}^{B}$,
$s^w \in \mathbb{R}^{N}$ are the scale vectors broadcast over the appropriate axes.

## Task

Implement `int8_linear(X, W)`:

```python
import numpy as np

def int8_linear(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    ...
```

**Inputs:**

- `X`: `float64` array of shape $(B, K)$ — activation matrix.
- `W`: `float64` array of shape $(N, K)$ — weight matrix.

**Output:**

- `float64` array of shape $(B, N)$ — dequantized result of the INT8 matmul.

**Rules:**

1. Use **NumPy only** — no PyTorch, no CUDA, no Python `for` loops over rows.
2. Quantized values must be `np.int8` and clamped to $[-128, 127]$.
3. The integer dot-product must accumulate in `int32` to avoid overflow.
4. If a row is all zeros (so its max-abs is $0$), set its scale to avoid division by zero.
5. The function must return `float64`.

## Example

```python
import numpy as np

X = np.array([[0.5, -0.3, 0.8],
              [0.1,  0.0, -0.7]])   # B=2, K=3
W = np.array([[0.2, -0.6,  0.4],
              [0.9,  0.1, -0.5]])   # N=2, K=3

Y = int8_linear(X, W)               # shape [2, 2], dtype float64
```

Each of the 2 token rows is independently scaled and quantized to `int8`.
Each of the 2 weight rows is independently scaled and quantized to `int8`.
The $2 \times 2$ output is the dequantized integer matmul.

## What the gate checks

One gate. A NumPy oracle applies the identical per-channel / per-token quantization
algorithm and computes the expected output $Y_{\mathrm{oracle}}$. The gate passes when

$$
\mathrm{rel\_err}
= \frac{\lVert Y_{\mathrm{learner}} - Y_{\mathrm{oracle}} \rVert_2}
        {\lVert Y_{\mathrm{oracle}} \rVert_2 + \epsilon}
< 10^{-3}
$$

A correct implementation should yield $\mathrm{rel\_err}$ near machine epsilon.
Errors in scale computation, clipping bounds, integer overflow, or wrong
quantization granularity will produce a large relative error and fail the gate.
