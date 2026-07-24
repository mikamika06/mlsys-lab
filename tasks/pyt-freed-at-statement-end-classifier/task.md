## Context

CPython primarily uses reference counting. When an object has no remaining strong references, its reference count reaches zero and the object can be destroyed immediately.

For a temporary object created during a statement, the important question is whether the statement leaves another reference behind. If the last reference disappears while evaluating the statement, the object is freed before the next statement. If a name, container, or other object keeps a reference, the object survives.

The classifier returns a boolean value:

$$
\mathrm{freed} =
\begin{cases}
\mathrm{True} & \text{if the temporary object is destroyed by the end of the statement} \\
\mathrm{False} & \text{if a reference keeps it alive}
\end{cases}
$$

The supplied statements call `make()` to create one tracked temporary object. The statement may store that object somewhere or may let it disappear immediately.

## Task

Implement `classify_freed(statement)`:

```python
def classify_freed(statement: str) -> bool:
    ...
```

The input is one Python statement string. Execute it in a controlled namespace containing a `make()` function. The function must return `True` exactly when the object returned by `make()` is freed by the time that statement finishes executing.

Examples of statements include direct calls, assignments, and container creation. Do not use text matching to classify statements. The result must be based on observing the object's lifetime.

The implementation may rely on CPython reference counting behavior.

## Example

```python
classify_freed("make()")
# True

classify_freed("x = make()")
# False

classify_freed("items = [make()]")
# False
```

## What the gate checks

The gate runs the function on multiple statements and compares the result with a CPython weak reference callback oracle. The oracle creates a weak-referenceable temporary object, records whether its callback runs, executes the statement, and uses the callback state to determine whether the temporary was freed.

The `exact_match` score must be exactly $1.0$.
