## Context

Low-bit linear layers reduce memory traffic by storing activations and weights in integer formats while keeping the output close to the original floating point computation.

In this task, the activation matrix $X \in \mathbb{R}^{m \times k}$ uses per-token int8 quantization. Each token row has one scale:

$$
s^x_i = \frac{\max_j |X_{ij}|}{127},
$$

and the quantized activation is

$$
X^q_{ij} = \mathrm{round}\left(\frac{X_{ij}}{s^x_i}\right).
$$

The weight matrix $W \in \mathbb{R}^{n \times k}$ uses int4 quantization with groups along the input dimension. For a group size $g$, each weight group has a scale:

$$
s^w_{r,t} = \frac{\max_{j \in G_t} |W_{rj}|}{7},
$$

where $G_t$ is one group of $g$ consecutive input columns. The quantized weight values are clipped to the int4 range:

$$
W^q_{rj} =
\mathrm{clip}\left(
\mathrm{round}\left(\frac{W_{rj}}{s^w_{r,t}}\right),
-8, 7
\right).
$$

The integer matrix multiplication uses the quantized tensors:

$$
Y^q_{ir} = \sum_j X^q_{ij} W^q_{rj}.
$$

The final output applies both dequantization scales:

$$
Y_{ir} = s^x_i \sum_j X^q_{ij} W^q_{rj} s^w_{r,t(j)} .
$$

Implementing this correctly requires preserving the per-token activation scales and the per-group weight scales.

## Task

Implement `linear_8da4w(X, W, group_size)`:

```python
def linear_8da4w(X: np.ndarray, W: np.ndarray, group_size: int) -> np.ndarray:
    ...
```

The inputs are:

- `X`: a float NumPy array with shape $(m, k)$.
- `W`: a float NumPy array with shape $(n, k)$.
- `group_size`: a positive integer that divides $k$.

Return a float64 NumPy array with shape $(m, n)$ containing the 8da4w quantized linear result. The implementation should quantize the activations per token, quantize the weights per group, perform the integer accumulation, and apply the scale factors.

## Example

```python
import numpy as np

X = np.array([[1.0, -2.0, 3.0, 1.0]])
W = np.array([[2.0, 1.0, -1.0, 4.0]])
Y = linear_8da4w(X, W, 2)
```

For `group_size=2`, the weight columns are split into two groups before computing the int4 scales.

## What the gate checks

The gate computes the 8da4w result independently using a NumPy oracle. It compares the submitted implementation output with the oracle using maximum absolute error:

$$
\mathrm{max\_abs\_err} = \max_{i,r}|Y_{ir}^{student} - Y_{ir}^{oracle}|.
$$

The result must satisfy $\mathrm{max\_abs\_err} \le 10^{-4}$ on multiple generated cases.
