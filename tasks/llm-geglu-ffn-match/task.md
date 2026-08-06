## Context

The GeGLU (Gated GELU) feed‑forward network is a variant of the standard MLP used in transformer architectures. It splits the linear projection into two parts, applies a non‑linearity to one part and multiplies element‑wise with the other:

$$
\text{output} = \operatorname{gelu}(X W_{\text{gate}})\;\odot\; (X W_{\text{up}})
$$

where $X$ is the input matrix of shape $(n, d_{\text{in}})$, $W_{\text{gate}}, W_{\text{up}}\in \mathbb{R}^{d_{\text{in}}\times d_{\text{out}}}$ are learnable weight matrices and $\odot$ denotes element‑wise product. The GELU activation is defined as

$$
\operatorname{gelu}(z) = \tfrac12 z \Bigl(1 + \tanh\!\bigl(\sqrt{\tfrac{2}{\pi}}\,(z+0.044715\,z^3)\bigr)\Bigr).
$$

Implementing this operation efficiently with Python requires only a few linear algebra calls and no explicit Python loops.

## Task

Define the function `geglu_ffn` that takes three arguments:

```python
def geglu_ffn(x: list[list[float]], w_gate: list[list[float]], w_up: list[list[float]]) -> list[list[float]]:
    ...
```

* `x` – a 2‑D array of shape `(batch, d_in)` with dtype float64.
* `w_gate`, `w_up` – weight matrices of shape `(d_in, d_out)` also float64.

The function must return the matrix of shape `(batch, d_out)` computed as described above. The output should be a list of dtype float64 and contain no NaNs or Infs.

## Example

```python

x = [[1., 2.], [3., 4.]]
w_gate = [[1.0 if i == j else 0.0 for j in range(2)] for i in range(2)]
w_up   = [[1.0] * 2 for _ in range(2)]

out = geglu_ffn(x, w_gate, w_up)
print(out)  # [[2.5235759718248305, 5.863793082263324], [20.97453825542759, 27.999508278362654]]
```

(The numbers come from applying the GELU formula to `x` and multiplying by a matrix of ones.)

## What the gate checks

The grader evaluates your implementation against a Python reference using the metric

$$
\max_{i,j} |\, \text{candidate}_{ij} - \text{reference}_{ij}\,|
$$

and requires this value to be at most $10^{-5}$.

Additionally, the function must run without raising exceptions on random inputs of size up to $(256, 512)$ for both input and output dimensions.
