## Context

CPython uses a Global Interpreter Lock (GIL) that prevents multiple threads from
executing Python bytecode simultaneously in one process. A workload that spends
most of its time in Python instructions usually does not get CPU scaling from
threads.

Some operations release the GIL while running native code. Examples include
many blocking I/O operations and computational kernels implemented by C
extensions. These workloads can overlap between threads.

A classifier can reason about a callable's behavior from its CPython bytecode and
the native operations it uses. The distinction is not based on elapsed time:
timing measurements are noisy and depend on the machine. The goal is to classify
the execution model.

## Task

Implement `classify_thread_scaling(workloads)`:

```python
def classify_thread_scaling(workloads):
    ...
```

`workloads` is a list of dictionaries. Each dictionary contains:

- `"name"`: a string identifier.
- `"fn"`: a Python callable representing the workload.

Return a dictionary mapping every workload name to a boolean.

Return `True` when the workload should scale across threads because its main work
can run while the GIL is released. Return `False` when it is dominated by Python
bytecode execution under the GIL.

The classifier should inspect the callable rather than execute long workloads.

## Example

```python
result = classify_thread_scaling([
    {"name": "python_loop", "fn": python_loop},
    {"name": "matrix_multiply", "fn": matrix_multiply},
])

# {
#   "python_loop": False,
#   "matrix_multiply": True,
# }
```

## What the gate checks

The gate creates workload descriptors containing real Python callables and checks
the returned mapping against a reference classifier.

The reference classifier uses CPython's `dis` module to inspect bytecode patterns
and checks native-operation behavior from real extension modules. The comparison
is exact: every workload name must have the correct boolean classification.
