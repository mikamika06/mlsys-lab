## Context

A running Python function executes inside a frame object. The tracing protocol exposed
by `sys.settrace` receives events from that frame, including `line` events. For a
line event, the frame attribute `f_lineno` is the source line number that Python is
about to execute.

For a function call with executed lines

$$L = (l_1, l_2, \dots, l_k),$$

the goal is to recover the ordered sequence of line numbers emitted by the tracing
protocol. This is different from listing all source lines because branches and
loops may cause the same line to appear multiple times.

## Task

Implement `predict_line_sequence(fn)`.

The function receives a zero-argument Python function and returns a list of integers.
The list must contain the `f_lineno` values observed for that function's `line` trace
events, in execution order.

Use the Python tracing protocol. Do not inspect source text or reconstruct execution
from bytecode metadata. The implementation should work for arbitrary zero-argument
functions that contain branches and loops.

The returned value should only include line events from `fn` itself, not helper
functions used by the tracer.

## Example

```python
def sample():
    x = 0
    for i in range(2):
        x += i
    return x

predict_line_sequence(sample)
# [2, 3, 4, 3, 4, 5]
```

The exact numbers in this example depend on where the function is defined. The
important property is that repeated loop iterations produce repeated line events.

## What the gate checks

The gate defines several fixture functions and uses CPython's tracing mechanism as
the oracle. It runs the candidate implementation on each fixture and compares the
returned integer sequence with the sequence collected from a real `sys.settrace`
trace.

The `exact_match` score must be $1.0$. A solution that returns source line metadata,
bytecode line starts, or a guessed control-flow path will not match the executed
trace.
