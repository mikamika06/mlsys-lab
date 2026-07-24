## Context

Python stores imported module objects in `sys.modules`. The first import of a
module name creates the module object and executes its top-level code. Later
imports of the same name reuse the cached object and skip top-level execution.

If a module's top-level execution count is stored outside the module namespace,
the number of executions after repeated imports reveals whether caching occurred.

For two imports of the same module name, correct caching gives:

$$
\mathrm{top\_level\_executions} = 1 .
$$

If the cached module entry is removed between imports, the module is loaded
again and the count becomes:

$$
\mathrm{top\_level\_executions} = 2 .
$$

## Task

Implement `cached_import_count(module_name, module_dir)`.

The function receives an importable module name and a directory containing that
module. It must:

1. Temporarily add `module_dir` to `sys.path`.
2. Remove any existing `module_name` entry from `sys.modules`.
3. Import the module using `importlib.import_module`.
4. Import the same module name a second time without clearing the cache.
5. Return the integer value of the module attribute `EXECUTIONS`.

Use Python's import system. Do not execute the module source manually.

## Example

A test module may contain:

```python
with open(COUNT_FILE, "w", encoding="utf-8") as f:
    value = int(f.read()) + 1 if f.read() else 1
```

The gate supplies a valid module that records its top-level executions in a
separate file. Calling:

```python
cached_import_count("counter_mod", "/tmp/example")
```

returns:

```python
1
```

because the second import reuses the same module object.

## What the gate checks

The gate uses the real CPython import machinery as the oracle. It creates a
temporary module whose top-level code records each execution externally, then
computes the expected execution marker using two normal
`importlib.import_module` calls.

The implementation must return exactly the same marker. Removing the module from
`sys.modules` before the second import causes the top-level code to run again and
fails the gate.
