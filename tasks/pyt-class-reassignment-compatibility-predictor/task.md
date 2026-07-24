## Context

Python objects have a runtime type, and class objects are themselves instances of metaclasses. This type/instance/metatype relationship determines many behaviors of the object model.

CPython supports changing an object's class at runtime with an assignment such as:

```python
obj.__class__ = OtherClass
```

This operation is restricted. The source and destination classes must have compatible instance layouts. For example, classes with incompatible slot layouts cannot be exchanged even if they are otherwise related through normal Python inheritance rules.

The compatibility condition is an implementation detail of CPython. The most direct way to know whether a reassignment works is to attempt the operation on a temporary instance and observe whether CPython accepts it.

## Task

Implement `predict_class_reassignment(pairs)`:

```python
def predict_class_reassignment(pairs):
    ...
```

The argument `pairs` is a list of `(SourceClass, TargetClass)` tuples. Return a boolean list with one entry per pair.

For each pair, the result must be `True` if CPython allows:

```python
instance.__class__ = TargetClass
```

where `instance` is a fresh instance of `SourceClass`. Return `False` if the assignment raises `TypeError`.

The function must not permanently modify the provided classes. Creating temporary objects is allowed.

## Example

```python
class A:
    pass

class B:
    pass

class C:
    __slots__ = ("value",)

answer = predict_class_reassignment([(A, B), (A, C)])
```

The returned booleans describe the actual CPython result of trying those assignments.

## What the gate checks

The gate creates classes with different CPython instance layouts and computes the expected answers by performing the real `__class__` assignment operation at runtime.

Your output is compared with this runtime oracle using exact boolean matching. The gate does not contain hand-written expected compatibility tables.
