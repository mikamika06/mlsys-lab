## Context

In Python a *closure* is a function that captures variables from its defining scope. The captured variable lives in a *cell*, and the inner function can read or modify it using the `nonlocal` keyword. This mechanism allows us to build stateful objects without resorting to classes.

A common pattern is a counter: each call increments an internal integer and returns the new value. Implementing such a counter with a closure requires that the inner function declares the captured variable as `nonlocal`.

## Task

Implement `make_counter(start=0)` that returns a callable. The returned callable, when invoked, should increment an internal counter starting from `start` and return the updated value.

```python
def make_counter(start: int = 0) -> Callable[[], int]:
    ...
```

The first call must return `start + 1`, the second `start + 2`, etc. The implementation must use a nonlocal cell variable; using global state or mutable default arguments is not allowed.

## Example

```python
>>> counter = make_counter(5)
>>> counter()
6
>>> counter()
7
>>> counter()
8
```

## What the gate checks

The grader creates two counters with different starting values, calls each one a fixed number of times and compares the produced sequence to that of a reference implementation. The result must match exactly; any deviation causes the gate to fail.
