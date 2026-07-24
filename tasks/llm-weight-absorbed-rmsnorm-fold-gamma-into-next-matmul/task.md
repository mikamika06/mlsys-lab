## Context

RMSNorm is a normalization layer used in many language models. For an input vector
$x \in \mathbb{R}^{d}$, RMSNorm without mean subtraction is

$$
\mathrm{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}},
$$

where $\gamma \in \mathbb{R}^{d}$ is a learned scale vector and $\odot$ denotes
elementwise multiplication.

A following linear layer computes

$$
z = W y + b,
$$

where $W \in \mathbb{R}^{m \times d}$ and $y=\mathrm{RMSNorm}(x)$.

The learned RMSNorm scale can be absorbed into the linear layer weights. Since the
RMS denominator depends only on the input vector, we can rewrite the computation as

$$
W\left(\gamma \odot \frac{x}{\mathrm{rms}(x)}\right)+b
=
(\gamma \odot W)\frac{x}{\mathrm{rms}(x)}+b.
$$

This removes the separate gamma multiplication during inference while preserving
the combined output.

## Task

Implement `fold_rmsnorm_gamma(W, b, gamma)`.

The function receives:

- `W`: a NumPy array of shape $(m,d)$ containing the next linear layer weights.
- `b`: a NumPy array of shape $(m,)$ containing the linear bias.
- `gamma`: a NumPy array of shape $(d,)$ containing the RMSNorm scale.

Return a tuple `(W_folded, b_folded)` where `W_folded` has the RMSNorm scale
absorbed into the input dimension of the weight matrix and `b_folded` is the
unchanged bias.

The folded layer must satisfy the same result as applying RMSNorm first and then
the original linear layer.

Use NumPy operations only.

## Example

```python
import numpy as np

W = np.array([[2.0, 3.0], [4.0, 5.0]])
b = np.array([1.0, -1.0])
gamma = np.array([10.0, 20.0])

W2, b2 = fold_rmsnorm_gamma(W, b, gamma)

# W2 is:
# [[20.0, 60.0],
#  [40.0, 100.0]]
#
# b2 is unchanged:
# [1.0, -1.0]
```

## What the gate checks

The gate creates random weights, biases, gamma vectors, and inputs. It computes the
reference output by applying RMSNorm followed by the original matrix multiplication
using NumPy. It then computes the output using the folded weights returned by the
candidate solution.

The maximum absolute difference

$$
\max_i |z_i^{\mathrm{folded}}-z_i^{\mathrm{reference}}|
$$

must be less than $10^{-5}$.
