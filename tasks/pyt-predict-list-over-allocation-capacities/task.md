## Context

CPython lists store elements in a dynamically allocated array. The number of slots
reserved in that array is called the allocated capacity. A list can grow without
reallocating until its current capacity is exhausted.

When appending an item that requires growth, CPython 3.12 computes a new capacity
using an over-allocation rule. For a requested list size $n$, the growth estimate
is

$$
\text{new\_allocated} = n + \left\lfloor\frac{n}{8}\right\rfloor + 6
$$

and the result is rounded down to a multiple of $4$ slots. The capacity sequence
describes how the internal storage changes as appends happen.

## Task

Implement `predict_list_capacities(n)`:

```python
def predict_list_capacities(n: int) -> list[int]:
    ...
```

The function receives a non-negative integer $n$ and returns the allocated
capacity after each append operation from an initially empty list through
$n$ appends. The returned list must contain one entry for each append that
causes an allocation change. Do not use `sys.getsizeof`, inspect CPython
internals, or create a list just to observe its capacity. Derive the sequence
from the growth rule.

## Example

```python
predict_list_capacities(10)
# [4, 8, 16]

predict_list_capacities(20)
# [4, 8, 16, 24]
```

## What the gate checks

The gate compares the returned sequence with a reference generated from the real
CPython list implementation. It creates a list, appends elements, and reads the
allocated capacity through `sys.getsizeof` and pointer-size accounting. The
solution must exactly match the observed capacity changes.
