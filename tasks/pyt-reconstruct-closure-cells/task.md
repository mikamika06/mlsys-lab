## Context

A Python function can capture variables from an enclosing scope. These captured values are stored in closure cells, which are exposed through the function attribute `__closure__`.

The names of captured variables are stored separately in the code object as `co_freevars`. If a function has a closure, the entries in `__closure__` correspond to those names by position.

For a function $f$, reconstruction means creating the mapping

$$
[(n_0, c_0), (n_1, c_1), \dots, (n_k, c_k)]
$$

where each $n_i$ is a closed-over variable name and each $c_i$ is the corresponding cell's `cell_contents`.

## Task

Implement `reconstruct_closure(fn)`:

```python
def reconstruct_closure(fn):
    ...
```

The function receives a Python function object and returns a list of pairs. Each pair must be:

```python
(name, value)
```

where `name` is a string from `fn.__code__.co_freevars` and `value` is the current contents of the matching closure cell.

If the function has no closure, return an empty list.

The order of pairs must match the order used by CPython in `co_freevars`.

## Example

```python
def make_adder(x):
    def add(y):
        return x + y
    return add

f = make_adder(10)

reconstruct_closure(f)
# [("x", 10)]
```

## What the gate checks

The gate creates nested functions with real CPython closures and computes the expected result directly from `__code__.co_freevars` and `__closure__`.

Your implementation passes when its returned list exactly matches the oracle result for all generated cases.
