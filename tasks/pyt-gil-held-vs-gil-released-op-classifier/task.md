## Context

The CPython Global Interpreter Lock (GIL) allows only one thread to execute Python bytecode at a time. However, some long-running operations implemented in C release the GIL while they wait for external work or perform native computation.

An operation can be viewed as a function $f$ that runs for some interval of time. A practical observation is whether another Python thread can make progress while $f$ is running. If another thread advances a counter during the call, the operation likely released the GIL for a meaningful portion of its execution.

This task uses a behavioral classifier. It does not inspect implementation details. Instead, it classifies operations by running them with a competing thread and observing whether concurrent progress occurs.

## Task

Implement `classify_gil_release(ops)`:

```python
def classify_gil_release(ops: list[str]) -> dict[str, bool]:
    ...
```

The input is a list of operation names. Return a dictionary mapping every operation name to `True` if the operation releases the GIL during its execution and `False` otherwise.

The supported operation names are:

- `"sleep"`: a `time.sleep` call.
- `"dot"`: a large NumPy matrix multiplication.
- `"hash"`: a large hashlib SHA-256 update.
- `"python_loop"`: a pure Python arithmetic loop.
- `"socket_recv"`: a blocking socket receive.

Use the behavior of the operation, not only its name. The function should return a mapping for all requested operations.

## Example

```python
result = classify_gil_release(["sleep", "python_loop"])

# Example shape:
# {
#     "sleep": True,
#     "python_loop": False
# }
```

## What the gate checks

The gate builds the real operations on the current interpreter and computes reference labels using a runtime concurrency oracle. The oracle starts a competing Python thread, runs each operation, and checks whether the competing thread made measurable progress while the operation executed.

The returned dictionary must exactly match the oracle labels for all fixture operations. The gate metric is:

$$
\mathrm{exact\_match} =
\frac{\text{number of correctly classified operations}}
{\text{number of tested operations}}.
$$

A value of $1.0$ is required.
