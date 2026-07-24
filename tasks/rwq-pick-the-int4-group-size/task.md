## Context

Group-wise int4 quantization gives every contiguous group of
`group_size` weights its own scale. Smaller groups track local
magnitude better (lower reconstruction MSE) but cost more: each
group's scale is itself stored (as fp16, 16 bits), so a smaller
`group_size` means more scales per weight — a real storage overhead
that competes with the accuracy gain.

For a symmetric int-`bits` quantizer with $q_{\max}=2^{\text{bits}-1}-1$,
a group $g$ reconstructs as

$$
\text{scale}(g) = \frac{\max(|g|)}{q_{\max}}, \qquad
\hat g_i = \text{scale}(g)\cdot\mathrm{clip}\!\big(\mathrm{round}(g_i/\text{scale}(g)),\, -q_{\max},\, q_{\max}\big)
$$

and the two competing quantities for a candidate `group_size` are

$$
\mathrm{mse}(\text{gs}) = \frac{1}{N}\sum_i \big(\hat W_i - W_i\big)^2, \qquad
\mathrm{overhead}(\text{gs}) = \frac{16}{\text{gs}} \;\text{bits/weight}
$$

(one fp16 scale amortized over `gs` weights). The combined cost trades
them off with a fixed weight $\lambda$:

$$
\mathrm{cost}(\text{gs}) = \mathrm{mse}(\text{gs}) + \lambda \cdot \mathrm{overhead}(\text{gs})
$$

Picking the best `group_size` means minimizing this cost over a set of
candidates — too small wastes bits on redundant scales, too large lets
outliers blow up every group's rounding resolution.

## Task

Implement `pick_int4_group_size`:

```python
def pick_int4_group_size(W: np.ndarray, group_sizes=(32, 64, 128, 256),
                          bits: int = 4, lam: float = 0.02):
    ...
```

- `W`: 1-D `float64` array. `len(W)` is divisible by every value in `group_sizes`.
- `group_sizes`: candidate group sizes to evaluate, in the given order.
- `bits`: quantizer bit width ($q_{\max}=2^{\text{bits}-1}-1$).
- `lam`: the overhead weight $\lambda$ above.

For every candidate `gs` in `group_sizes`: quantize `W` in contiguous
groups of `gs` elements (formulas above) and compute `cost(gs) =
mse(gs) + lam * (16.0 / gs)`.

Return `(best_group_size, best_cost, costs)`:
- `best_group_size`: the `int` from `group_sizes` with minimum cost
  (ties broken by earliest in `group_sizes`).
- `best_cost`: its `float` cost.
- `costs`: `float64` array of costs, one per candidate, in the same
  order as `group_sizes`.

## Example

```python
import numpy as np
W = np.random.default_rng(0).standard_normal(1024)
best_gs, best_cost, costs = pick_int4_group_size(W)
# costs[i] is the (mse + lam*overhead) cost of group_sizes[i]==32,64,128,256
# best_gs is whichever of those minimizes costs
```

## What the gate checks

The grader builds several seeded weight vectors and evaluates the exact
same cost formula independently in NumPy for every candidate
`group_size`.

`argmin_match` is `1.0` if your `best_group_size` equals the oracle's
argmin on every case, else `0.0` — this is exact-match (must equal
`1.0`), so picking the wrong candidate (e.g. always the smallest group
size, ignoring the overhead term entirely) fails immediately.
`cost_max_abs_err` is the worst-case absolute difference between your
`costs` array and the oracle's, across all candidates and cases (must
be `<= 1e-9`) — this catches a right *argmin* reached via a wrong cost
formula (e.g. missing the `lam` term, wrong overhead formula, or a
`qmax` off-by-one) that happened to still pick the correct candidate.
