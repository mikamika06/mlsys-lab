## Context

For a neural network layer we often have the activation vector at its input and the activation vector at its output for each sample in a batch. Let $x^{\text{in}}_{b,i}\in \mathbb R^d$ denote the input of layer $i$ for sample $b$, and $x^{\text{out}}_{b,i}$ its output. The cosine similarity between these two vectors is

$$
c_{b,i} = \frac{\langle x^{\text{in}}_{b,i},\,x^{\text{out}}_{b,i}\rangle}
              {\|x^{\text{in}}_{b,i}\|\;\|x^{\text{out}}_{b,i}\|}.
$$

The *block influence* of layer $i$ is defined as the expected deviation from perfect alignment:

$$
\mathrm{BI}_i = 1 - \mathbb E_b[c_{b,i}],
$$

where $\mathbb E_b$ denotes the average over all samples in the batch. A value close to zero means that the input and output are almost perfectly aligned, whereas a large value indicates that the layer significantly transforms its input.

The task is to compute the influence of every layer in a network and return a ranking of layers by their removability (larger $\mathrm{BI}_i$ → more removable).

## Task

Implement `block_influence_ranking(x_in, x_out)`:

```python
def block_influence_ranking(
    x_in: np.ndarray,
    x_out: np.ndarray
) -> tuple[np.ndarray, list[int]]:
    ...
```

`x_in` and `x_out` are 3‑D NumPy arrays of shape `(batch_size, n_layers, features)` containing the input and output activations for each sample. The function must return a pair:

* `influences`: a 1‑D array of length `n_layers` with the block influence scores (dtype `float64`);
* `ranking`: a list of layer indices sorted in **descending** order of influence.

The implementation must use only vectorised NumPy operations; no explicit Python loops over layers or samples. The returned array must have dtype `float64`.

## Example

```python
import numpy as np

batch, layers, feat = 4, 3, 5
rng = np.random.default_rng(0)
x_in  = rng.standard_normal((batch, layers, feat))
x_out = rng.standard_normal((batch, layers, feat))

influences, ranking = block_influence_ranking(x_in, x_out)

print(influences)   # e.g. [0.1234, 0.5678, 0.2345]
print(ranking)      # e.g. [1, 2, 0]  (layer 1 has the largest influence)
```

## What the gate checks

Two gates are applied:

* **rel_err** – The global relative L₂ error between the returned `influences` and a NumPy oracle must be at most $10^{-6}$.
* **spearman** – The Spearman rank correlation between the candidate ranking and the oracle ranking must equal 1.0 (i.e., the order is identical).

The grader recomputes the reference influence scores from scratch for each test case, so a correct implementation will satisfy both gates.
