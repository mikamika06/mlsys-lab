## Context

A transformer block composes a LayerNorm, an MLP sub-layer, and a residual
connection. Given input $x \in \mathbb{R}^{\text{batch} \times d}$, a single block
computes:

$$\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \beta$$

where $\mu = \frac{1}{d}\sum_i x_i$ and $\sigma^2 = \frac{1}{d}\sum_i (x_i - \mu)^2$
are the per-token mean and variance. The MLP sub-layer applies

$$\text{MLP}(\text{LN}(x)) = W_2 \, \text{ReLU}(W_1 \, \text{LN}(x) + b_1) + b_2$$

and the residual connection adds the original input back:

$$x' = x + \text{MLP}(\text{LN}(x))$$

Modern LLMs stack $N$ such blocks sequentially, **sharing** the same weight
matrices $(W_1, b_1, W_2, b_2)$ and norm parameters $(\gamma, \beta)$ across every
layer. After all $N$ blocks, a **final** LayerNorm with its own parameters
$(\gamma_f, \beta_f)$ is applied.

## Task

Implement the function `stack_blocks_forward`:

```python
def stack_blocks_forward(
    x, gamma, beta, W1, b1, W2, b2, gamma_f, beta_f, n_blocks
):
    """Apply N identical residual-MLP blocks then a final LayerNorm.

    Parameters
    ----------
    x       : np.ndarray, shape (batch, d)        – input embeddings
    gamma   : np.ndarray, shape (d,)               – block norm scale
    beta    : np.ndarray, shape (d,)               – block norm bias
    W1      : np.ndarray, shape (d, d_hidden)      – MLP first-layer weights
    b1      : np.ndarray, shape (d_hidden,)         – MLP first-layer bias
    W2      : np.ndarray, shape (d_hidden, d)       – MLP second-layer weights
    b2      : np.ndarray, shape (d,)                – MLP second-layer bias
    gamma_f : np.ndarray, shape (d,)               – final norm scale
    beta_f  : np.ndarray, shape (d,)               – final norm bias
    n_blocks: int                                    – number of blocks to stack

    Returns
    -------
    np.ndarray, shape (batch, d)  – output after N blocks + final norm
    """
```

Use $\varepsilon = 10^{-5}$ inside every LayerNorm. Work with `float64` throughout.
Use only NumPy — no Python loops over individual scalar elements. A loop over
the $N$ block applications is acceptable.

## Example

```python
import numpy as np

d, d_hidden, batch, n_blocks = 4, 8, 2, 3
np.random.seed(0)
x       = np.random.randn(batch, d)
gamma   = np.ones(d)
beta    = np.zeros(d)
W1      = np.random.randn(d, d_hidden) * 0.1
b1      = np.zeros(d_hidden)
W2      = np.random.randn(d_hidden, d) * 0.1
b2      = np.zeros(d)
gamma_f = np.ones(d)
beta_f  = np.zeros(d)

out = stack_blocks_forward(x, gamma, beta, W1, b1, W2, b2, gamma_f, beta_f, n_blocks)
# out.shape == (2, 4)
# Values differ from x because the MLP adds residual contributions through each block.
```

## What the gate checks

The grader runs five test cases with varying dimensions ($d \in \{4, 8, 16, 32\}$,
$d_\text{hidden} \in \{8, 16, 32, 64\}$, batch sizes 1–8, $N \in \{1, 2, 3, 5, 7,
10\}$). For each case it computes the reference output with its own LayerNorm +
MLP + residual implementation and reports

$$\text{max\_abs\_err} = \max_{\text{all cases}} \| \hat{y} - y \|_\infty$$

where $\hat{y}$ is your output and $y$ is the reference. The gate passes when
$\text{max\_abs\_err} < 10^{-4}$.
