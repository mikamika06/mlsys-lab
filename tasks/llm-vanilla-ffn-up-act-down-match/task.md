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
def ffn_forward(
    x: list[float],
    W_up: list[list[float]],
    b_up: list[float],
    W_down: list[list[float]],
    b_down: list[float],
) -> list[float]:
    """Compute a vanilla FFN forward pass with GELU activation.

    Parameters
    ----------
    x     : list[float], shape (d,)
    W_up  : list[list[float]], shape (d_hidden, d)
    b_up  : list[float], shape (d_hidden,)
    W_down: list[list[float]], shape (d, d_hidden)
    b_down: list[float], shape (d,)

    Returns
    -------
    list[float], shape (d,)
    """
```

Compute $y = W_{\text{down}}\,\text{GELU}(W_{\text{up}}\,x + b_{\text{up}}) + b_{\text{down}}$ using Python. Use the tanh approximation for GELU shown above. The weight matrices and biases are provided as arguments.

## Example

```python

d = 4
x     = [1.0, -1.0, 0.5, -0.5]
W_up  = [[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)]          # identity
b_up  = [0.0] * d
W_down = [[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)]
b_down = [0.0] * d

y = ffn_forward(x, W_up, b_up, W_down, b_down)
# Each entry y_i = GELU(x_i) since all matrices are identity and biases are zero.
# GELU(1.0) ≈ 0.8412, GELU(-1.0) ≈ -0.1588, etc.
```

## What the gate checks

The grader computes a reference output using the same tanh-approximation GELU formula and Python `@` operator on several test cases (fixed-seed random weights and inputs), then measures `max_abs_err` between your output and the reference. The worst error across all cases is reported.

The gate passes when `max_abs_err < 1e-5`. This checks that you applied the correct sequence of operations (linear $\to$ GELU $\to$ linear) with the exact GELU tanh approximation, and that your matrix-vector products use matrix multiplication rather than element-wise `*`.
