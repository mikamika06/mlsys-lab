## Context

CPython uses a cyclic garbage collector to find reference cycles that ordinary
reference counting cannot reclaim. Objects that can participate in reference
cycles are often GC-tracked, while some objects with no possible cyclic
references can be left untracked as an optimization.

The predicate exposed by CPython is `gc.is_tracked(obj)`. It returns a boolean
value:

$$
t(obj) =
\begin{cases}
1 & \text{if } obj \text{ is tracked by the cyclic GC} \\
0 & \text{otherwise}
\end{cases}
$$

The result depends on the object's actual runtime representation. For example,
mutable containers such as lists and many dictionaries are tracked, while some
immutable atomic objects are not. Some tuples containing only untracked objects
may also be optimized into an untracked state.

## Task

Implement `classify_gc_tracking(objects)`.

The function receives a list of arbitrary Python objects and must return a list
of booleans with the same length. Each output value must equal the result of
`gc.is_tracked` for the corresponding input object.

The function contract is:

```python
def classify_gc_tracking(objects):
    ...
```

Do not inspect object internals or use hardcoded type tables. Use the Python
garbage collection interface.

## Example

```python
import gc

objects = [1, [1, 2], {"x": 1}]
result = classify_gc_tracking(objects)

# result matches:
# [gc.is_tracked(1), gc.is_tracked([1, 2]), gc.is_tracked({"x": 1})]
```

## What the gate checks

The gate creates a set of representative Python objects and computes the
reference answer directly from CPython's `gc.is_tracked` oracle. The submitted
function must return an identical boolean list.

The metric is `exact_match`. Any incorrect assumption about which Python types
are tracked fails because tracking behavior is determined by the actual runtime
object state.
