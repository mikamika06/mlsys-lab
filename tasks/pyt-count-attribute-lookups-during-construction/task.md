## Context

Python object construction involves several protocol steps. Calling a class creates an instance by resolving construction methods on the type and then executing the class-defined `__new__` and `__init__` methods.

An attribute lookup performed by Python bytecode is represented by the `LOAD_ATTR` opcode. If a frame enables opcode tracing, the interpreter can report each executed opcode while a program runs.

For a construction call $obj = C()$, the number of observed attribute lookups depends on the implementation of the class and the methods executed during construction. This task measures those lookups using CPython's tracing interface rather than estimating them from source code.

## Task

Implement `count_attribute_lookups(cls)`:

```python
def count_attribute_lookups(cls):
    ...
```

The function receives a Python class object. It must construct an instance with `cls()` and return the number of `LOAD_ATTR` opcode events that occur during that construction.

Use `sys.settrace` with opcode tracing. The count must include attribute lookups executed while the construction protocol runs, including the class's `__new__` and `__init__` bodies.

The function should work for ordinary Python classes under CPython 3.12.

## Example

```python
class Example:
    def __new__(cls):
        obj = super().__new__(cls)
        obj.value = 10
        return obj

    def __init__(self):
        self.value = self.value + 1

count_attribute_lookups(Example)
# returns the number of LOAD_ATTR opcode events from Example()
```

## What the gate checks

The gate creates a real Python class with custom construction methods and uses CPython opcode tracing as the oracle. The returned value from `count_attribute_lookups` must exactly match the count produced by tracing the same construction protocol.

A solution that guesses from the source, uses bytecode inspection without executing construction, or returns a fixed value will fail because the gate derives the expected value from the running interpreter.
