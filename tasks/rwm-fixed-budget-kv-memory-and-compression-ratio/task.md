## Context

In many machine‑learning pipelines a key–value store is used to cache intermediate tensors or embeddings. The full memory footprint of such a store with $T$ entries and value dimension $d$ is
$$
\text{full}_\text{bytes}= T\, d\, \mathrm{sizeof}(\text{dtype}) + T\, \mathrm{sizeof}(\text{key}),
$$
where $\mathrm{sizeof}$ denotes the number of bytes required to represent a single element. When only a limited budget $B$ is available we must keep at most $B$ entries. A natural way to decide which entries survive is to retain those with the largest Euclidean norm, i.e.
$$
\|v_i\|_2 = \sqrt{\sum_{j=1}^{d} v_{ij}^2}.
$$

The compression ratio achieved by a particular selection is defined as
$$
\rho = \frac{\text{full}_\text{bytes}}{\text{kept}_\text{bytes}},
$$
where $\text{kept}_\text{bytes}$ counts only the bytes of the selected entries.

## Task

Implement `fixed_budget_kv(keys, values, budget)`:

```python
def fixed_budget_kv(keys: np.ndarray, values: np.ndarray, budget: int) -> dict:
    ...
```

`keys` is a 1‑D integer array of length $T$,
`values` is a 2‑D float array of shape $(T,d)$,
and `budget` is an integer $\le T$.  
The function must return a dictionary mapping each selected key to the
corresponding row in `values`. The selection criterion is the largest
$\ell_2$ norm among all rows, and at most `budget` entries are kept.
If `budget >= T`, all entries should be returned.

## Example

```python
import numpy as np
keys   = np.array([10, 20, 30])
values = np.array([[1., 0.],
                   [0., 2.],
                   [3., 4.]])          # norms: 1, 2, 5
B = 2
out = fixed_budget_kv(keys, values, B)
# out == {30: array([3., 4.]), 20: array([0., 2.])}
```

## What the gate checks

Two metrics are evaluated:

* **size_ratio_err** – the relative error between the compression ratio
  produced by your implementation and that of a reference oracle.
  The accepted value must satisfy `<= 1e-2`.

* **exact_keys** – a binary flag equal to `1.0` if the set of keys in the
  returned dictionary matches exactly the oracle’s selection, otherwise
  `0.0`. The gate requires equality.

The grader generates several random test cases with varying $T$, $B$ and
$d$.  Your solution must pass all gates on every case.
