## Context

Layer Normalization normalises each sample (row) of a batch independently.  
For an input matrix $X \in \mathbb{R}^{B\times D}$, the per‑sample mean and variance are

$$\mu_i = \frac{1}{D}\sum_{j=1}^D X_{ij}, \qquad
\sigma_i^2 = \frac{1}{D}\sum_{j=1}^D (X_{ij}-\mu_i)^2.$$

The normalised value is

$$\hat{X}_{ij} = \frac{X_{ij}-\mu_i}{\sqrt{\sigma_i^2+\varepsilon}},$$

where $\varepsilon$ is a small constant for numerical stability.  
Finally, learnable scale and shift parameters $\gamma,\beta \in \mathbb{R}^{D}$ are applied:

$$Y_{ij} = \gamma_j\,\hat{X}_{ij} + \beta_j.$$

The operation is performed row‑wise; all samples share the same $\gamma$ and $\beta$.

## Task

Implement a pure Python function that performs LayerNorm forward pass:

```python
def layer_norm(x: list[list[float]], gamma: list[float], beta: list[float]) -> list[list[float]]:
    ...
```

* `x` is a 2‑D array of shape `(batch, features)` and may be any numeric dtype.  
* `gamma` and `beta` are 1‑D arrays of length equal to the number of features.  
* The function must return an array of type `float64` with the same shape as `x`.  
* No explicit Python loops or list comprehensions are allowed; use vectorised Python operations only.

## Example

```python
from your_module import layer_norm

X = [[1.0, 2.0], [3.0, 4.0]]
gamma = [1.0, 1.0]
beta = [0.0, 0.0]

Y = layer_norm(X, gamma, beta)
print(Y)  # [[-0.9999800005999799, 0.9999800005999799], [-0.9999800005999799, 0.9999800005999799]]
```

## What the gate checks

The grader computes a reference implementation using Python and compares your output with it.  
It reports the maximum absolute difference

$$\max_{i,j} |Y^{\text{your}}_{ij}-Y^{\text{ref}}_{ij}|.$$

Your solution must achieve `max_abs_err <= 1e-6`.  Any deviation larger than this threshold will fail the gate.
