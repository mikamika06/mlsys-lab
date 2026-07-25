## Context

Every Python module object owns a namespace dictionary. A function defined at module scope stores a reference to that namespace as its global namespace:

$$
f.\texttt{\_\_globals\_\_} = M.\texttt{\_\_dict\_\_},
$$

where $M$ is the module containing the function definition.

When a function reads a global variable, Python looks up the name inside the dictionary referenced by the function's `__globals__` attribute. Mutating that dictionary changes the value visible to the function without changing the function object.

For a module-level function $f$, the identity relationship is:

$$
f.\texttt{\_\_globals\_\_} \text{ is } M.\texttt{\_\_dict\_\_} .
$$

## Task

Implement `module_globals_probe()`:

```python
def module_globals_probe():
    ...
```

The function must demonstrate the relationship between a function's global namespace and its module dictionary.

The function should:

1. Obtain the global namespace dictionary through `module_globals_probe.__globals__`.
2. Read the value stored under the key `"ARENA_GLOBAL"`.
3. Mutate that dictionary entry to `"mutated"`.
4. Read the global variable normally.
5. Return:

```python
(same_object, before, after)
```

where `same_object` is whether the function's `__globals__` dictionary is the same object used for its global lookup.

## Example

```python
ARENA_GLOBAL = "initial"

result = module_globals_probe()
# (True, "initial", "mutated")
```

The checker creates the global variable before calling the function, so the implementation should not rely on a literal value stored in the source file.

## What the gate checks

The gate uses real CPython function namespace behavior. It computes the expected tuple by running the same namespace mutation procedure on an independent module-level function, then compares the returned value.

A solution that returns constants or avoids mutating the module dictionary will not match the CPython behavior.
