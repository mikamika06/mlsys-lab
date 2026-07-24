## Context

Python imports are resolved through a chain of finders stored in `sys.meta_path`.
A finder can return a module specification, and a loader attached to that
specification can create and initialize a module object.

The import system creates module objects with metadata such as `__name__` and
`__spec__`. A virtual module can therefore expose behavior that depends on its
place in the import system even when no file exists.

The goal is to temporarily add a finder, let Python create the module through
normal import semantics, execute supplied source in the module namespace, and
read a value from the resulting namespace.

## Task

Implement `materialize_attr(name, source, attr)`:

```python
def materialize_attr(name: str, source: str, attr: str):
    ...
```

The function must:

1. Install a temporary finder on `sys.meta_path` that recognizes only the
   supplied module name.
2. Provide a loader that creates and initializes a module object.
3. Import the virtual module using `importlib.import_module`.
4. Execute the supplied source in the module namespace.
5. Return the value of `attr`.
6. Remove the temporary finder and temporary module entry after completion.

Do not create files on disk. The source code must run as imported module code, with
the normal import metadata available.

## Example

```python
value = materialize_attr(
    "virtual_math_mod",
    "answer = 40 + 2",
    "answer",
)
# value == 42
```

## What the gate checks

The gate uses an independent CPython import implementation as the reference
algorithm. It computes the expected values by installing a real
`MetaPathFinder`, creating modules through a loader, and importing them.

The cases include source code that reads import metadata such as `__name__` and
`__spec__.name`. The returned values must exactly match the oracle result for all
cases. The metric `exact_match` is $1.0$ only when every case succeeds.
