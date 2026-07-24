## Context

Python sets and dictionaries use hash tables, but their resize behavior is not identical. A set grows when its internal table becomes too full, and the resize points depend on the number of occupied entries rather than only on the number of user-visible elements.

For this task, the observable quantity is the allocation footprint reported by CPython. The function returns the sequence of `sys.getsizeof` values after each insertion into a set. This exposes resize events without relying on private C structures.

If $s_k$ is the set after $k$ successful additions, the returned sequence is

$$
[g_1, g_2, \dots, g_n], \quad g_k = \mathrm{sizeof}(s_k).
$$

The task uses a real CPython set as the oracle because implementation details such as table allocation are interpreter-specific.

## Task

Implement `set_allocation_trace(values)`:

```python
def set_allocation_trace(values):
    ...
```

The function receives an iterable of integer keys. Add the keys to a set in order and return a list containing the `sys.getsizeof` value of the set after every addition.

Duplicate keys still perform an insertion attempt but do not increase the number of elements in the set. The returned list must contain one entry per input value.

## Example

```python
trace = set_allocation_trace([1, 2, 3])
# trace is a list of CPython set allocation sizes after each add
```

The exact numbers are determined by the running CPython interpreter.

## What the gate checks

The gate builds a reference trace by performing the same operations on a real Python `set` and recording `sys.getsizeof` after every addition. The returned list must exactly match the oracle trace for several integer-key sequences.

A solution that models dictionary resizing instead of set resizing will fail because dictionary allocation behavior differs from set allocation behavior.
