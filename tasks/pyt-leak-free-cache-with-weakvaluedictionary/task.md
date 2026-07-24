## Context

A cache stores a mapping from keys to objects so repeated lookups can reuse existing values. A normal dictionary owns strong references to its values, which means the reference count of every cached object stays above zero while the entry exists. If the cache is long-lived, unused objects can remain reachable and create a memory leak.

A weak reference does not keep an object alive. Python's `weakref.WeakValueDictionary` removes an entry automatically when its value is garbage collected. The cache therefore only contains objects that are also referenced elsewhere.

If $C$ is a cache and $V$ is the set of external references to objects, a weak-value cache aims for the invariant

$$
\mathrm{keys}(C) = \{k \mid \mathrm{value}(k) \in V\}.
$$

The garbage collector may need to run before dead weak references are removed, so tests use explicit collection.

## Task

Implement `cache_surviving_keys(keep)`:

```python
def cache_surviving_keys(keep: list[int]) -> list[int]:
    ...
```

The function must:

1. Create cache entries with integer keys from `0` through `9`.
2. Store objects as values in a `weakref.WeakValueDictionary`.
3. Keep external references only for the keys listed in `keep`.
4. Force garbage collection before reading the remaining cache keys.
5. Return the surviving keys as a sorted list.

The returned list must contain only keys whose values are still alive outside the cache.

## Example

```python
print(cache_surviving_keys([1, 4, 7]))
# [1, 4, 7]

print(cache_surviving_keys([]))
# []
```

## What the gate checks

The gate builds the expected result using a real `weakref.WeakValueDictionary` and Python garbage collection behavior. Your implementation is compared against that oracle for several keep sets.

A cache implemented with a normal `dict` fails because the dictionary itself keeps strong references to all ten values, so deleted external references cannot remove entries.
