## Context

The IEEE 754 standard defines the product $0 \times \infty = \text{NaN}$ for
every signed combination of zero and infinity. This rule creates a subtle
failure mode when masking is implemented via multiplication.

Given a data vector $x \in \mathbb{R}^n$ and a boolean mask
$m \in \{0,1\}^n$, a naïve masked reduction is

$$\text{result} = \sum_{i=0}^{n-1} x_i \cdot m_i$$

When every $x_i$ is finite this works correctly: unmasked positions contribute
$0.0$. But when $x_i = \infty$ at a position where $m_i = \text{False}$,
the product $m_i \cdot x_i$ evaluates to $0 \times \infty = \text{NaN}$.
Floating-point addition is absorbing: once a single NaN enters the accumulator,
the entire sum becomes NaN — even though the offending infinity was supposed
to be excluded.

The standard fix replaces the multiply-mask with an explicit selection that
never forms the forbidden product. Two idiomatic Python alternatives are:

* **Fancy indexing** — `data[mask]` extracts only the selected elements before
  reduction, so masked-out values never participate.
* **Conditional selection** — substituting `0.0` (a finite
  value) for every unmasked position, producing a clean sum without any
  $0 \times \infty$ intermediate.

## Task

Fix the function `masked_sum` in `starter.py`:

```python
def masked_sum(data: list[float], mask: list[bool]) -> float:
    ...
```

It receives a 1-D `float64` array `data` and a 1-D boolean array `mask` of
the same length. It must return the sum of every `data[i]` where `mask[i]`
is `True`. NaN should appear in the output only when a *masked-in* value
(that is, at a position where `mask[i]` is `True`) is itself NaN.

The starter uses `data * mask` to zero out unmasked entries. Replace this
pattern with a NaN-safe alternative.

## Example

```python

data = [1.0, float('inf'), 3.0]
mask = [True, False, True]
masked_sum(data, mask)   # → 4.0   (not NaN!)
```

The buggy starter returns `NaN` because `False * inf` evaluates to
`0.0 * inf = NaN`.

## What the gate checks

One gate — **exact_match**. The returned float must equal the reference result
computed by `sum(d for d, m in zip(data, mask) if m)` across all test cases. The suite covers:
basic finite sums, masked-out infinities (the $0 \times \infty$ trap),
NaN propagation from masked-in values, mixed special values, and empty
reductions where every position is masked out.
