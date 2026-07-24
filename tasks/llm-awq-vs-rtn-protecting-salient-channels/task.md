## Context

AWQ (Activation-aware Weight Quantization) reduces the degradation of LLMs caused by quantizing salient weights. Round-To-Nearest (RTN) weight quantization ignores the fact that different input channels (activations) have different magnitudes. The quantization error on a highly salient channel gets multiplied by a large activation, severely corrupting the output.

AWQ protects salient channels by scaling up their weights before quantization, and inversely scaling them after. The per-channel scale $s_j > 1$ reduces the relative quantization error for that channel, shifting the rounding error to less important channels.

For a linear layer $Y = X W^\top$ where $W \in \mathbb{R}^{m \times n}$ and $X \in \mathbb{R}^{b \times n}$:
1. **Scale Weights**: $W'_{i,j} = W_{i,j} \cdot s_j$
2. **Quantize**: We apply symmetric INT4 quantization per row (output channel). For a row vector $v$:
   $$\Delta = \frac{\max_j |v_j|}{7} \quad (\text{if } \max_j |v_j| = 0, \text{ let } \Delta = 10^{-9})$$
   $$\hat{v} = \mathrm{clip}\left(\mathrm{round}\left(\frac{v}{\Delta}\right), -8, 7\right) \cdot \Delta$$
3. **Restore**: $\hat{W}_{i,j} = \hat{W}'_{i,j} / s_j$

## Task

Implement `compare_awq_rtn(W, X, s)`:

```python
import numpy as np

def compare_awq_rtn(W: np.ndarray, X: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    ...
```

It takes:
- `W`: float matrix of shape `(out_dim, in_dim)`
- `X`: float matrix of shape `(batch, in_dim)`
- `s`: float array of shape `(in_dim,)` representing the per-channel scaling factors.

Return `(err_rtn, err_awq)`, the relative L2 error (using the Frobenius norm) of the output matrix $Y = X \hat{W}^\top$ compared to the exact float output $Y_{\text{true}} = X W^\top$ for both:
1. `err_rtn`: Quantizing $W$ directly (RTN).
2. `err_awq`: Quantizing $W$ using the AWQ method with scales $s$.

The relative error between an approximation $Y_{\text{approx}}$ and $Y_{\text{true}}$ is:
$$\text{rel\_err} = \frac{\lVert Y_{\text{approx}} - Y_{\text{true}} \rVert_F}{\lVert Y_{\text{true}} \rVert_F}$$
where $\lVert \cdot \rVert_F$ is the standard `np.linalg.norm()`.

## Example

```python
import numpy as np

# A tiny layer
W = np.random.randn(4, 8)
X = np.random.randn(2, 8)

# Make channel 0 extremely salient
X[:, 0] *= 20.0
s = np.mean(np.abs(X), axis=0)**0.5

err_rtn, err_awq = compare_awq_rtn(W, X, s)
# Returns roughly: (0.12, 0.05)
```

## What the gate checks

The grader uses random arrays where a few channels have extremely high activation magnitudes. It tests that your implementation correctly yields `err_awq < err_rtn`, and matches the reference mathematical computation for both relative errors exactly.
