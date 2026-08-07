## Context

In many machine‑learning pipelines a tensor is stored in a sparse format, where only the non‑zero entries (the *live* parameters) are kept. The **density** of such a mask is defined as the fraction of zero elements relative to all elements:

$$\text{sparsity} = \frac{\#\text{zeros}}{\text{total elements}}
= 1 - \frac{\#\text{non-zeros}}{\text{total elements}}\,. $$

Knowing how many live parameters a mask contains and its sparsity is useful for memory budgeting, compression decisions, and runtime profiling.

## Task

Implement `count_live_params_and_sparsity(mask)`:

```python
def count_live_params_and_sparsity(mask: list[list[int]]) -> tuple[int, float]:
    ...
```

The function receives a 2‑D list of integers or booleans. It must return a tuple `(live_count, sparsity)` where `live_count` is the number of non‑zero entries (an `int`) and `sparsity` is a `float` in `[0,1]` representing the fraction of zero elements.

The implementation should use only Python operations; no explicit Python loops are required but allowed if you wish. The result must be computed with double precision (`float64`).

## Example

```python
mask = [[0, 1], [1, 0]]
live, sparsity = count_live_params_and_sparsity(mask)
print(live)      # 2
print(sparsity)  # 0.5
```

## What the gate checks

The grader generates several random masks and compares your output to a Python reference. The metric `exact_match` must equal `1.0`. A mismatch in either the count or sparsity causes the gate to fail.
