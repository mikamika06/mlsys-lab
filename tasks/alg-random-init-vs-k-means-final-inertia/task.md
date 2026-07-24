## Context

K‑means clustering partitions a set of $n$ points $\{x_i\}_{i=1}^n \subseteq \mathbb R^d$ into $k$ clusters by iteratively assigning each point to the nearest centre and recomputing centres as the mean of their assigned points.  
The quality of a final clustering is often measured by its **inertia** (within‑cluster sum of squared distances):

$$
\text{inertia} = \sum_{i=1}^{n}\lVert x_i - c_{\lambda(i)}\rVert^2,
$$

where $c_j$ denotes the centre of cluster $j$ and $\lambda(i)$ is the index of the nearest centre to $x_i$.  
A good initialisation can dramatically reduce the final inertia.  Two popular strategies are:

* **Random init** – choose $k$ points uniformly at random from the data.
* **K‑means++ init** – choose the first centre uniformly, then each subsequent centre with probability proportional to its squared distance from the nearest already chosen centre.

## Task

Implement `compare_inertia` that, for a given dataset and a list of seeds, returns the final inertia obtained by running K‑means twice per seed:

1. With **random** initialisation.
2. With **k‑means++** initialisation.

```python
def compare_inertia(X: np.ndarray,
                    n_clusters: int,
                    seeds: list[int]) -> tuple[np.ndarray, np.ndarray]:
    ...
```

* `X` – a 2‑D NumPy array of shape `(n_samples, n_features)`.
* `n_clusters` – the number $k$ of clusters.
* `seeds` – an iterable of integer seeds; for each seed you must run both initialisation strategies.

The function should return a tuple of two one‑dimensional `float64` arrays:

1. `random_inertias`: final inertia for each seed using random init.
2. `kpp_inertias`:   final inertia for each seed using k‑means++ init.

Both arrays must have length equal to `len(seeds)` and contain the inertias in the same order as the input seeds.

## Example

```python
import numpy as np
X = np.array([[0., 0.],
              [1., 0.],
              [0., 2.],
              [3., 4.]])
seeds = [42, 123]
random_inertias, kpp_inertias = compare_inertia(X, n_clusters=2, seeds=seeds)
print(random_inertias)   # e.g. array([5.25, 6.00], dtype=float64)
print(kpp_inertias)      # e.g. array([4.50, 5.75], dtype=float64)
```

The exact numbers depend on the implementation but should be reproducible for a given seed.

## What the gate checks

Two metrics are evaluated:

* **Relative error** – the global relative L2 error between the candidate’s output and a reference implementation must satisfy  
  $$\mathrm{rel\_err} \le 10^{-6}.$$

The reference is computed by running the same K‑means algorithm with NumPy only, so no external libraries are required. The gate will fail if the returned inertias differ from the reference beyond the tolerance.
