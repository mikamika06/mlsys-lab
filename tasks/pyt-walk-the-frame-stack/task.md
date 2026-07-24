## Context

Every executing Python function has a frame object. A frame stores execution state such as local variables and a link to the calling frame through `f_back`.

The call stack can be viewed as a sequence of frames:

$$
F_0 \leftarrow F_1 \leftarrow F_2 \leftarrow \dots \leftarrow F_n
$$

where each frame points to the frame that called it. Walking `f_back` moves from the current frame toward older callers.

Each frame contains a code object. The attribute `f_code.co_qualname` gives the qualified name of the function represented by that frame, including nesting information.

## Task

Implement `frame_qualname_chain()`:

```python
def frame_qualname_chain() -> list[str]:
    ...
```

The function must inspect the current Python frame and walk `f_back` repeatedly. Start from the caller frame (`f_back` of the function's own frame), and return the qualified names of frames in order from the nearest caller to the oldest reachable frame.

Do not use traceback formatting or inspect stack helpers. Use frame objects directly.

## Example

```python
def outer():
    def inner():
        return frame_qualname_chain()

    return inner()

chain = outer()
# The beginning of chain contains:
# ["outer.<locals>.inner", "outer", ...]
```

The exact outer frames depend on the runtime environment.

## What the gate checks

The gate calls the implementation at a controlled call site and compares the returned chain against a reference implementation that uses real CPython frame objects and the same `f_back` traversal rule.

The result must be an exact list of qualified frame names.
