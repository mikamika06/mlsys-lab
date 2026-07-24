## Context

Python dictionaries use a hash table with open addressing. The table stores indices into a compact entries array. To keep lookup performance stable, CPython grows the table before it becomes too full.

For a table with capacity $m$ and $n$ active entries, the growth threshold is based on a fill factor close to

$$
\frac{n}{m} = \frac{2}{3}.
$$

When the threshold is reached, CPython allocates a larger table and reinserts the existing entries.

The exact growth sequence is an implementation detail of CPython. This task asks you to observe and reproduce that sequence rather than reimplementing a simplified resizing formula.

## Task

Implement `dict_resize_sizes(keys)`:

```python
def dict_resize_sizes(keys: list[int]) -> list[int]:
    ...
```

Insert the integer keys from `keys` into a new dictionary in order. Return the list of internal dictionary table capacities after each insertion that causes a resize.

The returned values are table capacities such as $8$, $16$, or $32$, not the number of inserted keys and not the memory size in bytes.

The grader runs on CPython 3.12. Your implementation may inspect the real dictionary object representation to determine its current table capacity.

## Example

```python
sizes = dict_resize_sizes(list(range(20)))

# The exact sequence is CPython-version dependent.
# A valid result has the form:
# [initial growth size, next growth size, ...]
```

## What the gate checks

The gate builds a real CPython dictionary, inserts the same integer keys, and obtains the table capacity from the live dictionary object. The candidate output is compared with this oracle sequence.

The metric is `exact_match`. The solution must return exactly the same resize sequence for all tested insertion orders.
