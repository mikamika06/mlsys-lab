## Context

In a two‑layer feed‑forward network (FFN) the hidden neurons are connected to the input via an *up* projection matrix $U \in \mathbb{R}^{h\times d_{\text{in}}}$ and to the output via a *down* projection matrix $D \in \mathbb{R}^{d_{\text{out}}\times h}$.  
For each hidden neuron $i$ we can aggregate its incoming and outgoing weights into a single importance score by taking the Euclidean norm of the concatenated weight vector

$$
w_i = \bigl[\,U_{i,:},\; D_{:,i}\,\bigr] ,
\qquad
I_i = \| w_i \|_2 .
$$

This *group L2 norm* captures how strongly a neuron participates in the network: large incoming or outgoing weights both increase its importance.

## Task

Implement `per_neuron_importance(up_proj, down_proj)`:

```python
def per_neuron_importance(up_proj: np.ndarray,
                          down_proj: np.ndarray) -> np.ndarray:
    ...
```

* `up_proj` – 2‑D NumPy array of shape `(h, d_in)`; rows are the incoming weight vectors.  
* `down_proj` – 2‑D NumPy array of shape `(d_out, h)`; columns are the outgoing weight vectors.  
* Return a 1‑D float64 array of length `h`, where element `i` is
  $\sqrt{\sum_j U_{ij}^2 + \sum_k D_{ki}^2}$.

The implementation must use only NumPy vectorised operations (no Python loops).  
The output dtype must be `float64`.

## Example

```python
import numpy as np
up = np.array([[1, 0], [0, 2]])          # shape (2,2)
down = np.array([[3, 4], [5, 6]])        # shape (2,2)

imp = per_neuron_importance(up, down)
print(imp)   # [sqrt(1+9+25), sqrt(0+4+16+36)] ≈ [5.38516481, 7.21110255]
```

## What the gate checks

The grader computes a reference implementation with NumPy and compares your result using the relative L2 error

$$
\mathrm{rel\_err} = \frac{\|\,\hat I - I_{\text{ref}}\,\|}{\|\;I_{\text{ref}}\;\|}
$$

Your solution must satisfy $\mathrm{rel\_err}\le 10^{-6}$ on a set of random test cases. The returned array must also have the correct shape and dtype.
