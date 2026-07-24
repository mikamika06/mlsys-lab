## Context

Python modules are objects whose attributes live in a namespace dictionary. Reloading a module with `importlib.reload` does not create a new module object. Instead, Python re-executes the module's code while keeping the existing module identity and namespace.

A reload operation can be viewed as transforming a namespace $N$ by executing source code $S$ into the same object:

$$
N' = \mathrm{exec}(S, N)
$$

The object identity is preserved because the module object itself is reused. Attributes assigned by the new code are updated, while existing attributes may remain unless the new execution removes them explicitly.

## Task

Implement `reload_module_semantics(module, source)`:

```python
def reload_module_semantics(module, source):
    ...
```

The function receives a module-like object and a Python source string. Re-execute the source code in the existing module namespace.

Return a tuple:

```python
(module_id_unchanged, updated_attr)
```

where:

- `module_id_unchanged` is `True` only if the input module object is the same object after re-execution.
- `updated_attr` is the value of `module.value` after the source has executed.

The implementation should update the provided object's namespace rather than replacing the object with a new module.

## Example

```python
import types

m = types.ModuleType("demo")
m.value = 1

result = reload_module_semantics(m, "value = 42")

# result == (True, 42)
# m.value == 42
```

## What the gate checks

The gate creates real Python module objects and compares the candidate behavior against an oracle based on CPython's `importlib.reload` behavior.

The returned tuple must exactly match the oracle result for several module reload scenarios. A solution that creates a new module object, fails to update the namespace, or does not execute the supplied source will fail.
