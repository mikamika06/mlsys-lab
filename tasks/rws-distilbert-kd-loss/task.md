## Context

Knowledge distillation transfers knowledge from a large teacher model to a smaller student model.  
For a batch of $N$ examples with logits $\mathbf{t}\in\mathbb{R}^{N\times C}$ (teacher) and $\mathbf{s}\in\mathbb{R}^{N\times C}$ (student), the standard loss is

$$
L = \alpha\,T^{2}\,\mathrm{KL}\!\bigl(p_{\!t}\,\|\,p_{\!s}\bigr)
   + (1-\alpha)\,\mathrm{CE}\!\bigl(s, y\bigr),
$$

where  

* $p_{\!t}=\operatorname{softmax}\!\left(\frac{\mathbf{t}}{T}\right)$ and  
  $p_{\!s}=\operatorname{softmax}\!\left(\frac{\mathbf{s}}{T}\right)$ are the temperature‑scaled probability distributions,  

* $\mathrm{KL}(p_{\!t}\,\|\,p_{\!s}) = \displaystyle\frac1N
   \sum_{i=1}^{N}\sum_{c=1}^{C} p_{\!t,i,c}
   \bigl(\log p_{\!t,i,c}-\log p_{\!s,i,c}\bigr)$ is the mean Kullback–Leibler divergence,  

* $\mathrm{CE}(s,y) = -\,\frac1N
   \sum_{i=1}^{N} \log p_{\!s,i,y_i}$ is the cross-entropy between the student predictions and the true labels $y$.

The factor $T^{2}$ compensates for the temperature scaling of the KL term, ensuring that the two components are on a comparable scale.

## Task

Implement the function `kd_loss`:

```python
def kd_loss(
    teacher_logits: np.ndarray,
    student_logits: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.5,
    temperature: float = 1.0
) -> float:
    ...
```

* `teacher_logits`, `student_logits` are 2‑D NumPy arrays of shape `(N, C)` containing raw logits.  
* `labels` is a 1‑D array of length `N` with integer class indices in `[0, C-1]`.  
* Return the scalar loss as a Python float (or NumPy scalar) with dtype `float64`.

The implementation must use only vectorised NumPy operations; no explicit Python loops are allowed. Numerical stability should be handled by subtracting the per‑row maximum before exponentiation.

## Example

```python
import numpy as np

teacher_logits = np.array([[2.0, 1.0], [0.5, -0.5]])
student_logits = np.array([[1.5, 0.5], [0.3, -0.7]])
labels = np.array([0, 1])

loss = kd_loss(teacher_logits, student_logits, labels,
               alpha=0.8, temperature=2.0)
print(loss)   # e.g., 0.123456789
```

## What the gate checks

The grader computes a reference loss using NumPy and compares it to your output with the relative error metric

$$\mathrm{rel\_err} = \frac{\lVert L_{\text{cand}}-L_{\text{ref}}\rVert}
                           {\lVert L_{\text{ref}}\rVert + 10^{-12}}.$$

The gate requires $\mathrm{rel\_err}\leq 1\times10^{-6}$ on a set of random test cases. A fully vectorised implementation that follows the formula above will pass; any missing temperature scaling, incorrect weighting, or use of loops will cause the relative error to exceed the threshold.
