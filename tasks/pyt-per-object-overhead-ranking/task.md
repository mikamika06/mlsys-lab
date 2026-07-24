## Context

Every Python value is represented by an object with interpreter-managed metadata. A
CPython object has a header containing information such as its reference count and
type pointer. The total memory footprint reported by `sys.getsizeof` includes the
object header and any fixed instance storage, but does not include memory owned by
referenced objects.

For an object $x$, let $s(x)$ be the value returned by `sys.getsizeof(x)`. Ranking
objects by this value means sorting according to

$$
s(x_1) \leq s(x_2) \leq \dots \leq s(x_n).
$$

Objects with smaller interpreter overhead appear earlier in the ranking. This task
uses the running CPython interpreter as the source of truth because object sizes are
implementation details rather than language-level constants.

## Task

Implement `rank_object_types()`:

```python
def rank_object_types() -> tuple[str, ...]:
    ...
```

The function must create one instance of each of these built-in values:

- `int`
- `float`
- `tuple()`
- a small `str`
- a small `bytes`
- `bool`
- `None`
- `complex`

Return a tuple containing their type names ordered by increasing
`sys.getsizeof` value. If two objects have the same size, break ties by the
lexicographic order of their type names.

The returned names must be strings such as `"int"` or `"float"`, not the objects
themselves.

## Example

```python
result = rank_object_types()

# Example shape only. Exact order depends on the pinned CPython version.
# ("bool", "NoneType", "int", ...)
```

## What the gate checks

The gate builds the same objects and uses CPython's `sys.getsizeof` as the oracle.
The submitted function output must exactly match the oracle-generated ordering of
type names. No fixed size table is used because object layout can vary between
Python versions.
