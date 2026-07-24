## Context

In CPython, integers in the range $[-5,\;256]$ are cached for efficiency.  
When two integer objects with a value inside this interval are created separately,
they refer to the same object and `is` returns `True`.  
For values outside the cache each construction yields a distinct object and
`is` evaluates to `False`.

## Task

Implement the function `is_small_int(n: int) -> bool` that determines whether
the integer `n` would be interned by CPython. The function should return
`True` if two separately created integers with value `n` are identical
(`a is b`) and `False` otherwise.

```python
def is_small_int(n: int) -> bool:
    ...
```

The implementation must work for all Python 3.12+ integer values and should not rely on CPython‑specific introspection; it should be a pure Python solution that matches the described behaviour.

## Example

```python
>>> is_small_int(100)
True          # 100 lies in the cached range [-5,256]
>>> is_small_int(300)
False         # 300 is outside the cache
```

## What the gate checks

The grader evaluates a fixed set of integers and compares your output with the result obtained by checking whether `n` falls inside the CPython small‑int cache. The metric `exact_match` must equal `1.0`; any mismatch yields failure.
