## Context

A vanilla transformer FFN (feed-forward network) applies two linear projections with a nonlinear activation between them:

$$y = W_{\text{down}}\;\text{GELU}\!\bigl(W_{\text{up}}\,x + b_{\text{up}}\bigr) + b_{\text{down}}.$$

The "up" projection maps an input $x \in \mathbb{R}^{d}$ to a wider hidden dimension $d_h$ via the matrix $W_{\text{up}} \in \mathbb{R}^{d_h \times d}$ and bias $b_{\text{up}} \in \mathbb{R}^{d_h}$. The activation introduces nonlinearity. The "down" projection maps back to $\mathbb{R}^{d}$ via $W_{\text{down}} \in \mathbb{R}^{d \times d_h}$ and $b_{\text{down}} \in \mathbb{R}^{d}$. This is the standard MLP block in transformer decoders (GPT, LLaMA, etc.).

The GELU activation (Hendrycks & Gimpel, 2016) is defined as

$$\text{GELU}(x) = x\,\Phi(x),$$

where $\Phi(x)$ is the CDF of the standard normal distribution. The widely-used tanh approximation is

$$\text{GELU}(x) \;\approx\; \frac{1}{2}\,x\!\left(1 + \tanh\!\left(\sqrt{\frac{2}{\pi}}\;\bigl(x + 0.044715\,x^{3}\bigr)\right)\right).$$

## Task

Implement `ffn_forward`:

```python
def ffn_forward(x, W_up, b_up, W_down, b_down):
    """Compute a vanilla FFN forward pass with GELU activation.

    Parameters
    ----------
    x     : np.ndarray, shape (d,)
    W_up  : np.ndarray, shape (d_hidden, d)
    b_up  : np.ndarray, shape (d_hidden,)
    W_down: np.ndarray, shape (d, d_hidden)
    b_down: np.ndarray, shape (d,)

    Returns
    -------
    np.ndarray, shape (d,)
    """
```

Compute $y = W_{\text{down}}\,\text{GELU}(W_{\text{up}}\,x + b_{\text{up}}) + b_{\text{down}}$ using NumPy. Use the tanh approximation for GELU shown above. The weight matrices and biases are provided as arguments.

## Example

```python
import numpy as np

d = 4
x     = np.array([1.0, -1.0, 0.5, -0.5])
W_up  = np.eye(d)          # identity
b_up  = np.zeros(d)
W_down = np.eye(d)
b_down = np.zeros(d)

y = ffn_forward(x, W_up, b_up, W_down, b_down)
# Each entry y_i = GELU(x_i) since all matrices are identity and biases are zero.
# GELU(1.0) ≈ 0.8412, GELU(-1.0) ≈ -0.1588, etc.
```

## What the gate checks

The grader computes a reference output using the same tanh-approximation GELU formula and NumPy `@` operator on several test cases (fixed-seed random weights and inputs), then measures `max_abs_err` between your output and the reference. The worst error across all cases is reported.

The gate passes when `max_abs_err < 1e-5`. This checks that you applied the correct sequence of operations (linear $\to$ GELU $\to$ linear) with the exact GELU tanh approximation, and that your matrix-vector products use `@` or `np.dot` rather than element-wise `*`.
