## Context

The iterator protocol has two separate roles. An **iterable** implements
`__iter__`, which must return an **iterator** — an object implementing both
`__iter__` (usually returning `self`) and `__next__`. A `for` loop calls
`iter(obj)` once at the start, then calls `next()` on the result until it
raises `StopIteration`.

A common bug merges the two roles into one object: the class keeps its
iteration position (an index, say) as instance state, implements `__next__`
directly on itself, and makes `__iter__` return `self`. The first `for` loop
over such an object works fine — it drives the shared state from $0$ up to
exhaustion. But because `__iter__` handed back the *same* object instead of a
fresh iterator, a **second** `for` loop over the same object calls
`iter(obj)` again, which again returns `self` — whose position is already at
the end. The second pass silently produces nothing.

The fix is to keep the iterable and the iterator as separate objects: the
iterable's `__iter__` method constructs and returns a brand-new iterator
object (with its own fresh position counter) on every call, so the iterable
itself never accumulates state and can be iterated as many times as needed.

## Task

Fix the class `Squares`, provided below with the bug described above:

```python
class Squares:
    """Iterable over 0**2, 1**2, ..., (n-1)**2."""
    def __init__(self, n):
        self.n = n
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= self.n:
            raise StopIteration
        val = self.i * self.i
        self.i += 1
        return val
```

`Squares(n)` must remain iterable with `for x in Squares(n): ...`, yielding
$0^2, 1^2, \dots, (n-1)^2$ in order, **and** it must produce the exact same
full sequence every time it is iterated — `list(sq)` followed by another
`list(sq)` on the same instance must return the same result twice, not an
empty list the second time.

## Example

```python
sq = Squares(4)
list(sq)  # [0, 1, 4, 9]
list(sq)  # [0, 1, 4, 9]   (not [] — the same instance is restartable)
```

## What the gate checks

For several values of `n`, the grader creates one `Squares(n)` instance and
calls `list(...)` on it twice, comparing both results to the reference
`[i * i for i in range(n)]`. `exact_match` is $1.0$ only if **both** passes
return the full correct sequence. The buggy version above returns the
correct sequence on the first pass but an empty list on the second, since
`__iter__` keeps handing back the same exhausted object.
