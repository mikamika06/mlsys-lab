## Context

Python classes are objects, and every class object is itself an instance of a metaclass. The `isinstance` operation consults the class object's metaclass through the `__instancecheck__` protocol.

For a normal class $C$, the expression

$$\mathrm{isinstance}(x, C)$$

should agree with Python's type relationship rules. A custom metaclass can override `__instancecheck__`, but an incorrect implementation can make a type object report false answers.

The goal is to repair the type/metaclass/instance triangle so that a custom type object behaves like a truthful view of another real Python type.

## Task

Implement `make_truthful_type(name, accepted_type)`.

The function must return a new class object named `name`. The returned class must use a custom metaclass whose `__instancecheck__` result matches the real Python result of:

```python
isinstance(value, accepted_type)
```

for every value tested by the gate.

The returned object must be a type object that can be passed as the second argument to `isinstance`.

Example:

```python
TruthyInt = make_truthful_type("TruthyInt", int)

isinstance(3, TruthyInt)      # True
isinstance("3", TruthyInt)    # False
```

Do not replace `isinstance` globally. The returned type should contain the corrected metaclass behavior.

## Example

```python
TruthySeq = make_truthful_type("TruthySeq", (list, tuple))

results = [
    isinstance([], TruthySeq),
    isinstance((), TruthySeq),
    isinstance({}, TruthySeq),
]

# [True, True, False]
```

## What the gate checks

The gate builds several truthful types and compares their `isinstance` tables with the result produced by CPython's own `isinstance(value, accepted_type)` operation.

The metric is `exact_match`. A passing solution must produce a table with no mismatches.
