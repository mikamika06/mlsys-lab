## Context

A Python traceback is a linked structure of `types.TracebackType` objects. Each node stores a frame, a source line number, a bytecode instruction offset, and a pointer to the next traceback node.

The chain direction is from the oldest frame to the newest frame when displayed by Python. A traceback can be assembled manually by creating nodes and connecting them through the `tb_next` field.

A traceback walk produces a sequence of observations:

$$
[(\mathrm{name}_0, \mathrm{line}_0), (\mathrm{name}_1, \mathrm{line}_1), \dots, (\mathrm{name}_{k-1}, \mathrm{line}_{k-1})]
$$

where each pair describes one frame in the traceback chain. The task is about reconstructing this structure, not raising a new exception.

## Task

Implement `reconstruct_traceback(frames)`.

`frames` is a list ordered from the oldest frame to the newest frame. Each element is a pair:

```python
(frame_object, line_number)
```

where `frame_object` is a live Python frame object and `line_number` is the line that should be stored in the traceback node.

The function must:

1. Build a synthetic `types.TracebackType` chain matching the given order.
2. Return the traceback walk as a list of `(code_name, line_number)` pairs.

The returned list must contain the code object name from each frame and the stored traceback line number.

## Example

```python
import sys

def inner():
    return sys._getframe()

def outer():
    outer_frame = sys._getframe()
    inner_frame = inner()
    return [(outer_frame, outer_frame.f_lineno),
            (inner_frame, inner_frame.f_lineno)]

frames = outer()

walk = reconstruct_traceback(frames)
# [
#   ("outer", <outer line number>),
#   ("inner", <inner line number>)
# ]
```

## What the gate checks

The gate creates real CPython frame objects and uses `types.TracebackType` itself to construct the reference traceback chain. The returned walk from the submitted implementation must exactly match the walk produced by the CPython traceback construction algorithm.

Returning only the input frames, using a reversed order, or creating independent nodes without linking `tb_next` correctly will fail.
