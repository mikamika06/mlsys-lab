## Context

A Python frame object represents an executing function call. A frame exposes the
current local namespace through `f_locals`, and its code object describes
compiler-created details such as closure variables.

When a nested function references a variable from an enclosing function, Python
stores that variable in a cell. These names are available from the frame's code
object through `co_cellvars`, and their current values can also appear in
`f_locals`.

A tracing function installed with `sys.settrace` receives frame events during
execution. A trace hook can inspect a frame at a particular line after local
assignments have happened and read the values stored by the call protocol.

## Task

Implement `read_frame_locals_and_cellvars()`.

The function must use a trace hook to observe an internal target function call.
At the trace point, return a dictionary with this exact structure:

```python
{
    "locals": {
        "number": 42,
        "text": "frame",
        "cell_value": 7
    },
    "cellvars": {
        "cell_value": 7
    }
}
```

The returned values must come from the captured frame. Do not call the target
function directly and inspect its return value. The implementation should read
the frame's `f_locals` and the cell variable names from the frame code object.

## Example

```python
result = read_frame_locals_and_cellvars()

assert result["locals"]["number"] == 42
assert result["locals"]["text"] == "frame"
assert result["cellvars"]["cell_value"] == 7
```

## What the gate checks

The gate creates a CPython frame oracle using `sys.settrace`. The oracle captures
the target frame at the same trace point and computes the expected result from
`f_locals` and `co_cellvars`.

The submitted implementation is executed and its returned dictionary is compared
with the oracle result using exact equality. A solution that guesses values or
fails to capture the frame through tracing does not pass.
