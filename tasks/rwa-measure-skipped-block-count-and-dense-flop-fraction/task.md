## Context

Block-sparse attention divides the $n_q \times n_{kv}$ pairwise score matrix
into a grid of $B_q \times B_{kv}$ blocks, each of size
$(\text{block\_size}_q \times \text{block\_size}_{kv})$.
A binary block mask $M \in \{0,1\}^{B_q \times B_{kv}}$ records which blocks
are computed ($M_{ij}=1$) and which are skipped ($M_{ij}=0$).

Two summary statistics capture how much work the sparsity pattern saves:

**Skipped-block count.** The total number of zero entries in $M$:

$$\text{skipped} = \sum_{i=1}^{B_q}\sum_{j=1}^{B_{kv}} \mathbf{1}[M_{ij} = 0]
  = B_q \cdot B_{kv} - \sum_{i,j} M_{ij}.$$

**Dense-FLOP fraction.** The fraction of the dense attention FLOPs that are
actually executed, equal to the proportion of blocks computed:

$$\text{flop\_fraction} = \frac{\sum_{i,j} M_{ij}}{B_q \cdot B_{kv}}.$$

When $\text{flop\_fraction} = 0$ every block is skipped and the attention costs
nothing; when $\text{flop\_fraction} = 1$ the result is identical to dense
attention.

## Task

Implement `measure_block_sparsity`:

```python
import numpy as np

def measure_block_sparsity(block_mask: np.ndarray) -> tuple[int, float]:
    ...
```

**Input.** `block_mask` — a 2-D boolean NumPy array of shape
$(B_q, B_{kv})$. A `True` entry means the corresponding block is computed.

**Output.** A tuple `(skipped_block_count, dense_flop_fraction)` where

- `skipped_block_count` is the `int` count of `False` entries in `block_mask`,
- `dense_flop_fraction` is the `float` ratio of `True` entries to total entries,
  i.e. $\text{flop\_fraction}$ above.

Handle the edge case $B_q \cdot B_{kv} = 0$ by returning `(0, 1.0)`.

Use vectorised NumPy operations. Do not write a Python double loop.

## Example

```python
import numpy as np

mask = np.array([[True,  False, True ],
                 [False, False, True ]])
skipped, frac = measure_block_sparsity(mask)
# skipped == 3,  frac == 0.5
```

## What the gate checks

Two metrics, each verified on six test cases (all-True, all-False, mixed,
single-element True, single-element False, and a seeded $8 \times 12$ random
mask):

| Metric | Condition | Meaning |
|---|---|---|
| `exact_match` | $= 1$ | Student's `skipped_block_count` equals the oracle integer on every case. |
| `flop_fraction_accuracy` | $\le 1$ | Student's `dense_flop_fraction` has relative error $\le 10^{-9}$ versus the oracle on every case. |

The oracle is computed by NumPy inside the grader itself — no values are
hardcoded.
