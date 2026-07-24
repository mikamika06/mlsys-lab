## Context

Creating an object in Python involves a two-stage construction protocol. The class method
`__new__` creates and returns the instance, while `__init__` initializes an instance
after creation. Method lookup follows the type hierarchy, so overriding either method
changes which implementation runs.

For a class hierarchy with a base class $B$ and subclasses $S_1, S_2, \dots$, the
construction event sequence is an ordered list

$$L = [(c_1, m_1), (c_2, m_2), \dots, (c_k, m_k)]$$

where each pair records the class name $c_i$ whose method ran and the method name
$m_i \in \{\_\_new\_\_, \_\_init\_\_\}$.

The `type` object is involved in object creation because classes are themselves
instances of metaclasses. However, ordinary instance construction still follows the
`__new__` then `__init__` protocol defined by the class hierarchy.

## Task

Implement `construction_sequence()`.

The function must create the class hierarchy and construct objects so that it returns
the exact ordered log of all `__new__` and `__init__` calls. The returned value must
be a list of two-item tuples:

```python
[
    ("ClassName", "__new__"),
    ("ClassName", "__init__"),
]
```

The implementation should demonstrate the normal Python construction protocol. The
grader will provide no inputs and will call the function directly.

## Example

A valid implementation may return a sequence produced by constructing several classes:

```python
[
    ("Base", "__new__"),
    ("Child", "__new__"),
    ("Child", "__init__"),
]
```

The exact sequence depends on the hierarchy and objects constructed by the function.

## What the gate checks

The gate creates an independent instrumented class hierarchy at grading time and uses
Python's actual object construction behavior as the oracle. The result returned by
`construction_sequence()` is compared with the sequence produced by that runtime
instrumentation.

The metric is `exact_match`. A score of $1.0$ requires the returned ordered event list
to exactly match the reference construction sequence.
