## Context

In Python, a property is a descriptor that allows attribute access to be managed by getter/setter methods. A common pattern is to cache the result of an expensive computation so that subsequent accesses are cheap. However, sometimes we need the value to be recomputed on every access—for example when the underlying data can change or when side effects must occur each time.

## Task

Implement a class `RecomputeCounter` that stores a sequence of numbers and exposes a read‑only property `sum`. The property must **recompute** the sum from scratch on every access; it may not cache the result. The implementation should contain exactly one executable line in the getter body so that each access executes that line once.

```python
class RecomputeCounter:
    def __init__(self, data):
        ...
```

The constructor receives an iterable of numbers and stores them internally. The `sum` property must return the arithmetic sum of those numbers.

## Example

```python
from your_module import RecomputeCounter

rc = RecomputeCounter([1, 2, 3])
print(rc.sum)   # 6
print(rc.sum)   # 6 again, but recomputed each time
```

## What the gate checks

The grader uses `sys.settrace` to count how many times the body of the property’s getter is executed when the property is accessed five times. The expected count is exactly **5**; any caching or multi‑line implementation will produce a different number and fail the gate.
