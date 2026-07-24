## Context

In knowledge distillation a *teacher* network produces logits $z_t$ and a *student* network produces logits $z_s$.  
To compare the two predictive distributions we use the Kullback–Leibler (KL) divergence.  
With temperature scaling $T>0$ the softened probabilities are

$$
p_i = \frac{\exp(z_{t,i}/T)}{\sum_j \exp(z_{t,j}/T)},\qquad
q_i = \frac{\exp(z_{s,i}/T)}{\sum_j \exp(z_{s,j}/T)} .
$$

The forward KL (teacher $\Vert$ student) and the reverse KL (student $\Vert$ teacher) are

$$
D_{\mathrm{KL}}(p\|q)=\sum_i p_i\,\log\frac{p_i}{q_i}, \qquad
D_{\mathrm{KL}}(q\|p)=\sum_i q_i\,\log\frac{q_i}{p_i}.
$$

Both terms are used in training objectives and diagnostics.

## Task

Implement the function `kl_divergences`:

```python
def kl_divergences(
    teacher_logits: np.ndarray,
    student_logits: np.ndarray,
    temperature: float
) -> Tuple[float, float]:
    ...
```

The function receives one‑dimensional NumPy arrays of logits and a positive temperature.  
It must return a tuple `(forward_kl, reverse_kl)` containing the two KL divergences as Python `float`s (or NumPy scalars).  
All computations should be vectorised with NumPy; no explicit Python loops are required.

## Example

```python
import numpy as np
teacher = np.array([2.0, 1.0, 0.1])
student = np.array([1.5, 1.2, 0.3])
T = 2.0
fwd, rev = kl_divergences(teacher, student, T)
print(f"forward: {fwd:.6f}, reverse: {rev:.6f}")
```

## What the gate checks

The grader evaluates the returned pair against a NumPy reference implementation on several random test cases.  
It reports the maximum relative L2 error

$$
\mathrm{rel\_err} = \frac{\lVert \hat y - y\rVert}{\lVert y\rVert + 10^{-12}},
$$

where $y$ is the reference pair and $\hat y$ the candidate output.  
The solution must satisfy $\mathrm{rel\_err}\le 1\times10^{-6}$.
