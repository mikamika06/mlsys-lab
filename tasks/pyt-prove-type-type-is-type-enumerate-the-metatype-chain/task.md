## Context

In Python, every object has a type, and type objects also have types. The
relationship forms a metaclass chain.

For an object $x$, the first element is its class:

$$c_0 = \mathrm{type}(x).$$

The next elements are found by repeatedly taking the type of the previous
element:

$$c_{i+1} = \mathrm{type}(c_i).$$

The chain reaches a fixed point when the name of the current type is `type`.
This demonstrates the special role of `type` in Python's object model:
`type(type)` is `type` itself.

## Task

Implement `metatype_chain(x)`:

```python
def metatype_chain(x):
    ...
```

The function must return a list of strings containing the names of the classes
in the metatype chain.

Start with `type(x).__name__`, then repeatedly continue with
`type(current_type).__name__` until the chain reaches the fixed point
`"type"`. Include the final `"type"` entry exactly once.

The function must work for normal instances, classes, built-in objects, and
objects with custom classes.

## Example

```python
class Demo:
    pass

obj = Demo()

metatype_chain(obj)
# ["Demo", "type"]

metatype_chain(Demo)
# ["type", "type"]
```

## What the gate checks

The gate builds several real Python objects and computes the expected chains by
walking CPython's actual `type(...)` relationships at grading time. Your output
must exactly match the oracle list of type names for every object.

The metric `exact_match` is the fraction of tested objects whose returned chain
matches the oracle.
