## Context

Attention in transformer models is often analyzed with the roofline model, which
compares the number of operations performed with the amount of data moved from
memory.

For a sequence length $n$ and embedding dimension $d$, scaled dot-product
attention has three main stages:

$$
S = QK^\top
$$

$$
P = \mathrm{softmax}(S)
$$

$$
O = PV
$$

where $Q$, $K$, and $V$ each have shape $n \times d$, while $S$ and $P$ have
shape $n \times n$.

Arithmetic intensity (AI) is defined as

$$
AI = \frac{\text{FLOPs}}{\text{bytes moved}}.
$$

For this task, use a simple roofline estimate with float32 tensors. Count a
multiply-add as two FLOPs and assume each stage reads its inputs and writes its
output once. The stages are:

$$
AI_{QK} =
\frac{2n^2d}{(2nd + n^2)\cdot 4}
$$

$$
AI_{softmax} =
\frac{5n^2}{(2n^2)\cdot 4}
$$

$$
AI_{PV} =
\frac{2n^2d}{(2nd + n^2)\cdot 4}
$$

The softmax estimate includes subtraction of the row maximum, exponentiation,
sum, and normalization as five elementwise operations per score.

## Task

Implement `attention_ai(seqlen, dim)`:

```python
def attention_ai(seqlen: int, dim: int) -> np.ndarray:
    ...
```

Return a NumPy array of shape `(3,)` containing the arithmetic intensity for:

1. The $QK^\top$ matrix multiplication.
2. The softmax stage.
3. The $PV$ matrix multiplication.

The returned values must be `float64`.

## Example

```python
import numpy as np

x = attention_ai(1024, 64)
# array containing:
# [AI for QK^T, AI for softmax, AI for PV]
```

## What the gate checks

The gate computes the expected arithmetic intensities from the formulas using a
NumPy reference implementation. The submitted implementation is compared using
relative error:

$$
\mathrm{rel\_err}
=
\frac{\lVert x_{\mathrm{candidate}}-x_{\mathrm{reference}}\rVert}
{\lVert x_{\mathrm{reference}}\rVert + 10^{-12}}
$$

The value must satisfy $\mathrm{rel\_err} \le 10^{-6}$ across several sequence
length and dimension combinations.
