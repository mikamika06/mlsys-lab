## Context

The CPython GIL allows only one thread to execute Python bytecode at a time. Real GIL behavior depends on the interpreter implementation, operating system scheduling, extension modules, and blocking operations, so this task uses a deterministic model instead of measuring wall-clock execution.

A workload is represented as a sequence of operations. Each operation has a modeled effect:

- `compute` represents Python bytecode execution that keeps the GIL.
- `alloc` represents a Python allocation path that keeps the GIL but creates a scheduling boundary in this model.
- `io` represents a blocking operation that releases and reacquires the GIL.

The model counts GIL ownership transitions. Let $R$ be the number of releases and $A$ be the number of acquisitions. The reported event count is

$$
E = R + A .
$$

The first operation starts while the current thread already owns the GIL, so no initial acquire is counted. Every `io` operation causes one release and one later acquire. Every `alloc` operation causes one modeled release and acquire pair after the allocation completes. `compute` operations do not change ownership.

## Task

Implement:

```python
def modeled_gil_count(ops: list[str]) -> int:
    ...
```

The function receives a list containing only the strings `"compute"`, `"alloc"`, and `"io"`. Return the modeled number of GIL acquire/release events as an integer.

Do not inspect interpreter internals or use timing. The function should implement the deterministic model described above.

## Example

```python
ops = ["compute", "io", "compute", "alloc"]

modeled_gil_count(ops)
# 4
```

The `io` operation contributes one release and one acquire. The `alloc` operation contributes one release and one acquire, giving $2 + 2 = 4$ modeled events.

## What the gate checks

The gate generates several mixed workloads and compares the result against an independently implemented model of the same deterministic GIL transition rules.

The returned integer must exactly match the reference count. The gate does not measure real thread scheduling because real GIL timing is not deterministic across platforms.
