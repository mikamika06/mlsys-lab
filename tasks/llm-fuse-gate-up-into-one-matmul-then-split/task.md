## Context

In many modern neural architectures a *gated* feed‑forward network is used.  
A popular variant is the **SwiGLU** (Swish‑Gated Linear Unit) where two linear
transforms are applied to an input vector $x$:

$$
u = xW_{\text{up}} + b_{\text{up}}, \qquad
g = xW_{\text{gate}} + b_{\text{gate}},
$$

the gate is passed through the swish activation

$$
\operatorname{swish}(z) = \frac{z}{1+\exp(-z)},
$$

and finally the two outputs are multiplied element‑wise:

$$
y = u \odot \operatorname{swish}(g).
$$

A naïve implementation performs **two** matrix multiplications followed by a
non‑linear activation and an element‑wise product.  This can be fused into a
single matrix multiplication if we concatenate the weight matrices and biases:

$$
W = [\, W_{\text{up}} \;\; W_{\text{gate}} \,], \qquad
b = [\, b_{\text{up}} \;\; b_{\text{gate}} \,].
$$

Then

$$
z = xW + b,
$$

and the first half of $z$ is $u$, the second half is $g$.  Applying swish to the
second half and multiplying with the first gives the same result as above but
with only one expensive matmul.

## Task

Implement `fused_swiglu` that performs this fused operation:

```python
def fused_swiglu(x, w_up, b_up, w_gate, b_gate):
    ...
```

* `x`: 2‑D NumPy array of shape `(n, d)`  
* `w_up`, `w_gate`: weight matrices of shape `(d, h)`  
* `b_up`, `b_gate`: bias vectors of shape `(h,)`  

The function must return a NumPy array of shape `(n, h)` with dtype `float64`
containing the SwiGLU output.  Only vectorised NumPy operations are allowed;
no explicit Python loops.

## Example

```python
import numpy as np

x = np.array([[0., 1.], [2., 3.]])
w_up   = np.ones((2, 4))
b_up   = np.zeros(4)
w_gate = -np.eye(2, 4)
b_gate = np.full(4, 0.5)

y = fused_swiglu(x, w_up, b_up, w_gate, b_gate)
print(y)
```

The output will be a `(2, 4)` array of the SwiGLU activations.

## What the gate checks

* **max_abs_err** – The maximum absolute difference between your result and
  a NumPy reference implementation must not exceed $10^{-6}$.
* **op_count** – Using `sys.settrace` the grader counts how many Python line
  events are executed inside your function.  This count must be at most 50,
  encouraging a single‑matmul implementation.

A correct solution will pass both metrics; a broken one (e.g. missing swish or
performing two separate matmuls) will fail at least one of them.
