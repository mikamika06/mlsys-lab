## Context

A tracing system captures a program as a directed acyclic graph (DAG) of operations instead of executing only the final result. Each node records an operation, its inputs, and the produced temporary value. This is the basic idea behind graph capture systems such as Python bytecode tracers and compiler frontends.

For a simple expression

$$z = (a + b) \times c$$

the graph contains an addition node followed by a multiplication node:

$$
t_0 = a + b,\qquad t_1 = t_0 \times c .
$$

The graph structure keeps dependencies explicit. A later compiler can inspect these dependencies, transform them, or generate another implementation.

In this task, a minimal tracer works from Python bytecode. The bytecode instruction stream describes operations and the evaluation stack describes how values flow between operations.

## Task

Implement `trace_function(fn)`:

```python
def trace_function(fn):
    ...
```

The function receives a Python function containing a small arithmetic expression and returns a list of DAG nodes.

Each node must be a tuple:

```python
(op, args, out)
```

where:

- `op` is a string operation name.
- `args` is a tuple of input names or previous temporary outputs.
- `out` is the produced temporary name.

The tracer must emit nodes for arithmetic operations only. Ignore load instructions and the final return instruction.

Supported operations:

- `+` becomes `"add"`
- `-` becomes `"sub"`
- `*` becomes `"mul"`
- `/` becomes `"truediv"`

Temporary outputs must be named in execution order as `"t0"`, `"t1"`, and so on.

Input variables are represented by their Python argument names. Constants are represented as `"const:<value>"`.

## Example

```python
def expr(x, y):
    return (x + y) * 2

trace_function(expr)
```

returns:

```python
[
    ("add", ("x", "y"), "t0"),
    ("mul", ("t0", "const:2"), "t1")
]
```

## What the gate checks

The gate builds a reference graph from real CPython bytecode using the `dis` module. The reference is generated from the function's actual instructions and stack behavior, not from hardcoded expected outputs.

The returned DAG is canonicalized and must exactly match the bytecode-derived oracle for several arithmetic functions. A tracer that only handles one expression shape or emits incorrect dependency names will fail.
