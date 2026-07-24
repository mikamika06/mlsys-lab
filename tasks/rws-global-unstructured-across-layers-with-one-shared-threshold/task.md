## Context

`torch.nn.utils.prune.global_unstructured` with the `L1Unstructured` method prunes a
whole model by one shared magnitude threshold, computed over **every parameter tensor
combined** — not one threshold per tensor. This matters because layers can live at very
different weight scales: a layer initialized or trained to have tiny weights (e.g.
scale $\approx 10^{-3}$) would be almost entirely wiped out by a per-tensor threshold
tuned to a layer at unit scale, while a large-scale layer (e.g. scale $\approx 50$) would
barely be touched by that same per-tensor rule. A single **global** threshold instead
asks: across the whole model, which weights are smallest in absolute value, regardless
of which tensor they live in?

Given tensors $W_1, \dots, W_L$ with total element count $N = \sum_l |W_l|$, and a
target fraction `amount`:

$$
k = \mathrm{round}(\texttt{amount} \cdot N).
$$

Concatenate $|W_1|, \dots, |W_L|$ into one flat array of $N$ magnitudes, and prune
(mask out) the $k$ globally smallest of them — wherever they happen to live. Ties
(equal magnitudes) are broken by earliest position in the concatenation order (tensors
in the given list order, row-major within each tensor).

## Task

Implement `global_unstructured_masks`:

```python
def global_unstructured_masks(weights: list[np.ndarray], amount: float) -> list[np.ndarray]:
    ...
```

- `weights`: a list of `float64` arrays, arbitrary (and possibly different) shapes.
- `amount`: fraction, in $[0, 1]$, of the **total** element count across all tensors to
  prune globally.

Return a list of boolean arrays, one per input tensor (same shapes), `True` where the
weight is kept and `False` where it is pruned. Exactly $k = \mathrm{round}(\texttt{amount}
\cdot N)$ elements are `False` in total, across the whole list — the $k$ smallest
magnitudes anywhere among all the tensors.

## Example

```python
import numpy as np

w0 = np.array([0.01, 0.02])   # tiny scale
w1 = np.array([10.0, 20.0])   # large scale
masks = global_unstructured_masks([w0, w1], amount=0.5)
# total elements = 4, k = round(0.5*4) = 2
# the 2 globally smallest magnitudes are 0.01 and 0.02 -- both in w0
# masks[0] = [False, False]   (all of w0 pruned)
# masks[1] = [True,  True]    (w1 untouched, despite also being "amount"-eligible)
```

## What the gate checks

The gate builds a NumPy oracle running the identical global-threshold algorithm on a
fixed set of four layer tensors, deliberately generated at four very different
magnitude scales, at a fixed `amount`. It checks:

- `masks_exact_match`: every one of your returned masks must exactly match the
  oracle's, for every tensor (must be `1.0`).
- `sparsity_off_by`: the difference between the total number of elements you pruned
  (summed across all tensors) and the oracle's target $k$, must be at most `1`.
