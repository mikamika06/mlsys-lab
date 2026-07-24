## Context

Rotary Position Embedding (RoPE) injects relative positional information by rotating pairs of embedding dimensions.  
For a vector $x \in \mathbb{R}^d$ with even $d$, split it into $\frac d 2$ consecutive pairs $(x_{2i}, x_{2i+1})$.  
A rotation by angle $\theta_i$ is applied to each pair:

$$
\begin{pmatrix}
x'_{2i}\\
x'_{2i+1}
\end{pmatrix}
=
\begin{pmatrix}
\cos \theta_i & -\sin \theta_i\\
\sin \theta_i &  \cos \theta_i
\end{pmatrix}
\begin{pmatrix}
x_{2i}\\
x_{2i+1}
\end{pmatrix}.
$$

The angles are linear in the token position $p$ and a learned frequency $\omega_i$:  
$\displaystyle \theta_i = p\,\omega_i$.  This simple operation preserves dot‑product structure while encoding positional cues.

## Task

Implement `apply_rope(x, pos)`:

```python
def apply_rope(x: np.ndarray, pos: int) -> np.ndarray:
    ...
```

`x` is a 2‑D NumPy array of shape $(n,d)$ with even $d$.  
The function must return a new array of the same shape and dtype `float64`.  
For each pair of columns $(2i,2i+1)$ compute the rotation angle

$$
\theta_i = \text{pos} \times \omega_i,\qquad 
\omega_i = \frac{0.01 + 0.98\, i}{d/2 - 1},\quad i=0,\dotsc,\tfrac d 2-1.
$$

The implementation must be fully vectorised; no explicit Python loops over rows or pairs.

## Example

```python
import numpy as np
x = np.array([[1., 0., 0., 1.],
              [0., 1., 1., 0.]])
# d=4 → two frequency values: ω₀≈0.01, ω₁≈0.99
out = apply_rope(x, pos=2)
print(out)
```

The output will be a rotated version of `x` with the specified angles.

## What the gate checks

Two metrics are evaluated:

* **max_abs_err** – the maximum absolute difference between your result and a NumPy reference implementation. The solution must satisfy $\mathrm{max\_abs\_err} \le 10^{-6}$.
* **shape_and_dtype** – the output shape must match the input shape and the dtype must be `float64`.

The gate will fail if either condition is not met.
