## Context

Python lists and sets both support the `in` operator, but they use different
internal strategies.

A list stores values in order. A membership test must compare against elements
until it finds a match or reaches the end. For $n$ elements, the number of
comparisons can grow as

$$O(n).$$

A set stores hashed values in a table. After computing a hash, it usually jumps
directly to a small number of candidate entries. The expected lookup cost is

$$O(1).$$

This task measures the difference using equality operations. Integer values are
wrapped in objects that count how many times `__eq__` is called. This observes
real CPython membership behavior instead of estimating complexity.

## Task

Implement `membership_op_counts(keys, queries)`:

```python
def membership_op_counts(keys: list[int], queries: list[int]) -> tuple[int, int]:
    ...
```

The function must:

1. Create hashable probe objects containing the integers from `keys`.
2. Test every query against a Python `list` of those objects.
3. Test every query against a Python `set` of those objects.
4. Return:

```python
(list_equality_calls, set_equality_calls)
```

The returned values must be the number of equality calls triggered by the two
membership loops.

Do not approximate the counts. Use a real `__eq__` counter.

## Example

```python
list_calls, set_calls = membership_op_counts(
    [0, 1, 2],
    [2, 3]
)

# list_calls is larger because the list scans elements.
# set_calls is smaller because hashing avoids most comparisons.
```

## What the gate checks

The grader creates its own equality-counting probe class and computes the
reference result by running real CPython list and set membership operations.

The `list_eq_match` gate checks that the returned list comparison count matches
the oracle.

The `set_eq_match` gate checks that the returned set comparison count matches
the oracle.

A solution that implements the set measurement with a list scan will return the
wrong number of equality operations and fail.
