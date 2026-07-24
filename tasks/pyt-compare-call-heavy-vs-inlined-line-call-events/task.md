## Context

CPython's frame evaluation protocol creates a new frame object, and fires a
`'call'` trace event, every time control enters a Python function — whether
that function is your top-level entry point or a tiny one-line helper called
once per loop iteration. `sys.settrace` lets you observe this directly: a
tracer callback receives `('call', frame, None)` on every function entry and
`('line', frame, None)` on every bytecode line boundary executed. For an
identical numerical result, computing it through many small function calls
("call-heavy") versus inlining the same arithmetic directly in the loop body
produces wildly different `'call'` event counts even though the two pieces of
code compute the same number.

For a loop over $N$ elements: a call-heavy version that invokes one helper
function per element fires roughly $N + 1$ `'call'` events (one per helper
invocation, plus one for entering the outer function itself). An inlined
version that never calls anything from inside the loop fires exactly $1$
`'call'` event — for entering the outer function — regardless of $N$.

## Task

Implement both:

```python
def sum_squares_call_heavy(x: np.ndarray) -> float:
    ...

def sum_squares_inlined(x: np.ndarray) -> float:
    ...
```

Both must return $\sum_i x_i^2$ for a 1-D array `x`.

* `sum_squares_call_heavy` — loop over `x` and, for **each element**, call a
  separate Python helper function (e.g. `def _square(v): return v * v`) to do
  the squaring. One helper call per element, no vectorized shortcuts.
* `sum_squares_inlined` — loop over `x` and compute `v * v` directly in the
  loop body. No function call happens per element (calling `float()`,
  builtins, or numpy ufuncs directly on scalars does not create a traced
  Python-level `'call'` frame the way calling a `def`-defined function does —
  but to keep this unambiguous, simply don't define or call any per-element
  helper `def` at all).

## Example

```python
import numpy as np

x = np.array([1.0, 2.0, 3.0, 4.0])
sum_squares_call_heavy(x)   # 30.0
sum_squares_inlined(x)      # 30.0  -- same result, very different call trace
```

## What the gate checks

The grader runs both of your functions on the same 200-element array,
verifies the output against `np.sum(x**2)` (`rel_err`), and traces each call
with its own `sys.settrace` callback, counting `'call'` events separately for
each function:

* `call_heavy_events` — number of `'call'` events while running
  `sum_squares_call_heavy`. Gate: `>= 150` (one call-heavy run on 200
  elements fires roughly 201 — one helper call per element, plus entering the
  function itself).
* `call_inline_events` — number of `'call'` events while running
  `sum_squares_inlined`. Gate: `<= 10` (only entering the function itself —
  no per-element helper calls).

Both correctness and the structural call-count gap must hold simultaneously.
