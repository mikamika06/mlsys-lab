## Context

**Static batching** groups a fixed set of requests together and pads every
one of them, along the token axis, out to the length of the longest member
of that batch — a single dense tensor of shape
$(\text{batch\_size}, \max(\text{len}))$ is much simpler to compute on than
a ragged one, but every padding position is compute and memory spent on
nothing.

For a batch $b$ with request lengths $L_b = \{\ell_1, \dots, \ell_{n_b}\}$,
the number of token *slots* actually allocated is
$\max(L_b)\cdot n_b$, of which

$$
\text{wasted}_b = \max(L_b)\cdot n_b - \sum_{\ell \in L_b} \ell
$$

are pure padding. Across a set of batches, the overall wasted fraction is

$$
\text{wasted\_fraction} = \frac{\sum_b \text{wasted}_b}{\sum_b \max(L_b)\cdot n_b} .
$$

## Task

Implement `padding_waste_fraction`:

```python
def padding_waste_fraction(lens: np.ndarray, batch_ids: np.ndarray) -> float:
    ...
```

* `lens` — 1-D int array, the total (prompt + generation) token length of
  every request.
* `batch_ids` — 1-D int array, the same length as `lens`; `batch_ids[i]`
  identifies which static batch request `i` belongs to (batch ids need not
  be contiguous or sorted).

Return the scalar `wasted_fraction` defined above, computed over ALL
batches present in `batch_ids`.

## Example

```python
import numpy as np

lens = np.array([10, 8, 4, 20, 20])
batch_ids = np.array([0, 0, 0, 1, 1])

frac = padding_waste_fraction(lens, batch_ids)
# batch 0: max=10, n=3 -> slots=30, used=10+8+4=22 -> wasted=8
# batch 1: max=20, n=2 -> slots=40, used=20+20=40  -> wasted=0
# frac = (8 + 0) / (30 + 40) = 8 / 70
```

## What the gate checks

A single gate, **rel_err**, compares your returned scalar against a
reference computed directly from the same formula on a fixed fixture
(`lens.npy`, `batch_ids.npy` — 8 batches of 4-16 requests each, random
lengths). Must match to a relative error `<= 1e-9` (this is an exact
closed-form computation — no numerical tolerance is warranted beyond
floating-point noise).
