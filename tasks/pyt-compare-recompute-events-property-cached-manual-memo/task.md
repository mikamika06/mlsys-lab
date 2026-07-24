## Context

A Python `@property` is a descriptor: every attribute access runs its getter
function again, from scratch. If the getter does real work, that work is
repeated on every access — for $K$ reads, the underlying computation runs $K$
times.

`functools.cached_property` is also a descriptor, but a non-data one: the
*first* access runs the getter and then stores the result directly in the
instance's `__dict__` under the same attribute name. Because a plain
attribute in `__dict__` takes priority over a non-data descriptor during
lookup, every access after the first finds the cached value directly and
never calls the getter again — for $K$ reads, the computation runs once.

Manual memoization reproduces the same one-computation behavior by hand: the
getter checks an instance attribute, computes and stores the value only if it
has not been computed yet, and returns the stored value on every later call.

For a getter that runs an expensive computation $f(n)$, the number of times
$f$ actually executes across $K$ attribute reads is:

$$
\text{plain property}: K, \qquad
\text{cached\_property}: 1, \qquad
\text{manual memo}: 1 .
$$

## Task

Implement all four names below:

```python
def expensive(n: int) -> int:
    """Sum of squares 0^2 + 1^2 + ... + (n-1)^2, via an explicit loop."""
    ...

class PropertyDemo:
    """.value is a plain @property that calls expensive(self.n) every access."""
    def __init__(self, n: int): ...

class CachedPropertyDemo:
    """.value is a functools.cached_property wrapping expensive(self.n)."""
    def __init__(self, n: int): ...

class ManualMemoDemo:
    """.value is a property that manually memoizes expensive(self.n)."""
    def __init__(self, n: int): ...
```

`expensive(n)` must be a **module-level** function (not a method, not a
lambda bound at class-definition time) that computes the sum of squares with
an explicit `for` loop — do not use a closed-form formula. Each class takes
`n` in its constructor and stores it; its `.value` property must call the
module-level `expensive` function to obtain the result. `PropertyDemo` uses a
plain `@property` that recomputes on every access. `CachedPropertyDemo` uses
`@functools.cached_property`. `ManualMemoDemo` uses a plain `@property` whose
body checks a private instance attribute and only calls `expensive` the first
time, caching the result for later accesses.

## Example

```python
d = PropertyDemo(10)
d.value  # calls expensive(10)
d.value  # calls expensive(10) again

c = CachedPropertyDemo(10)
c.value  # calls expensive(10)
c.value  # returns the cached result, no call
```

## What the gate checks

The grader first verifies `expensive(n)` returns the correct sum of squares
for several values of `n`. It then wraps the module-level `expensive`
function with a real counting wrapper (a monkey-patch, not a self-reported
counter) and, for a fresh instance of each class, reads `.value` exactly
$K=5$ times, recording how many of those reads actually triggered a call to
`expensive`. It also checks that the final `.value` for each class equals the
correct sum of squares.

The three recorded counts must exactly match the reference vector
$[K, 1, 1] = [5, 1, 1]$ for `exact_match` to be $1.0$. An implementation that
caches inside `PropertyDemo`, or fails to cache inside `CachedPropertyDemo`
or `ManualMemoDemo`, produces a different count vector and fails the gate.
