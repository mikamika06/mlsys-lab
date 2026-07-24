## Context

A two‑layer fully connected network with a ReLU non‑linearity is defined by
$$
z = W_1x + b_1,\qquad a = \max(0,z),\qquad y = W_2a + b_2,
$$
where $W_1\in\mathbb{R}^{h\times d}$, $b_1\in\mathbb{R}^h$, $W_2\in\mathbb{R}^{o\times h}$ and $b_2\in\mathbb{R}^o$.
The gradient of a scalar loss $L(y)$ with respect to the input $x$ is
$$
\frac{\partial L}{\partial x}
  = W_1^\top \bigl(\mathbf{1}_{z>0}\odot (W_2^\top\,\nabla_y L)\bigr),
$$
where $\mathbf{1}_{z>0}$ is a binary mask that equals $1$ when the corresponding element of $z$ is positive and $0$ otherwise, and $\odot$ denotes element‑wise multiplication.

Gradient checkpointing discards intermediate activations during the forward pass and recomputes them in the backward pass.  For this simple network the only activation to discard is the ReLU output $a$.  The pre‑activation $z$ can be recomputed from $x$, $W_1$ and $b_1$ when needed.

## Task

Implement two functions that realise a checkpointed forward and backward pass for the network above:

```python
def checkpoint_forward(x: np.ndarray,
                       W1: np.ndarray, b1: np.ndarray,
                       W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """Return y = W2·ReLU(W1·x + b1) + b2 without storing intermediate activations."""
```

```python
def checkpoint_backward(dy: np.ndarray,
                        x: np.ndarray,
                        W1: np.ndarray, b1: np.ndarray,
                        W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """Given the upstream gradient dy = ∂L/∂y, return ∂L/∂x using only the input
    and weights (no cached activations)."""
```

Both functions must use NumPy only; no Python loops are allowed.  The forward pass should not keep any intermediate results that would normally be needed for back‑propagation.

## Example

```python
import numpy as np

# dimensions
d, h, o = 4, 5, 3

x   = np.random.randn(d)
W1  = np.random.randn(h, d)
b1  = np.random.randn(h)
W2  = np.random.randn(o, h)
b2  = np.random.randn(o)

y  = checkpoint_forward(x, W1, b1, W2, b2)          # forward
dy = np.random.randn(o)                            # upstream gradient

dx = checkpoint_backward(dy, x, W1, b1, W2, b2)      # backward
```

## What the gate checks

Two metrics are evaluated:

* **output_error** – the maximum absolute difference between the output of `checkpoint_forward` and a reference implementation that stores all activations.
* **grad_error** – the maximum absolute difference between the gradient returned by `checkpoint_backward` and the analytic gradient computed from the same network.

Both errors must be ≤ $10^{-5}$ for the solution to pass.  The grader generates several random test cases, computes the reference forward and backward passes analytically, and compares them with your implementation using NumPy’s `max_abs_err`.
