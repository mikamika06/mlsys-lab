## Context

Python lists are dynamic arrays. Appending an item usually places it in the next free slot of the allocated buffer, so the number of existing elements moved is zero.

Inserting at position $0$ requires every existing element to move one position to the right. If a list has length $k$ before an insertion, that insertion shifts $k$ elements. Repeating the operation for $n$ elements gives the modeled memory movement

$$
0 + 1 + 2 + \dots + (n-1) = \frac{n(n-1)}{2}.
$$

This is why repeated front insertion has $O(n^2)$ element movement, while repeated append has $O(n)$ list updates and zero element shifts.

## Task

Implement `shift_counts(n)`:

```python
def shift_counts(n: int) -> tuple[int, int]:
    ...
```

The function must return a pair:

- the modeled number of element shifts caused by performing `n` repeated `insert(0, value)` operations on a list,
- the number of element shifts caused by performing `n` repeated `append(value)` operations.

The second value is always the number of existing elements moved during append operations, not the number of reallocations.

Do not use the closed-form expression directly. Model the operations as they happen by tracking the current list length.

## Example

```python
shift_counts(5)
# (10, 0)
```

For the first result, the insert operations move:

$$
0 + 1 + 2 + 3 + 4 = 10
$$

existing elements in total.

## What the gate checks

The gate computes the expected result by simulating the list operation model independently and compares it with the returned pair.

The returned value must exactly match the oracle for several input sizes. Solutions that return only the asymptotic complexity or use the append count for both cases will fail.
