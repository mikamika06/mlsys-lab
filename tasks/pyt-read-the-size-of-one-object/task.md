## Context

Every Python object is represented by a structure managed by the CPython runtime. A
normal object has a universal object header containing metadata such as a reference
count and a pointer to its type object. The memory reported for an object includes
the bytes occupied by the object representation itself, but not necessarily all
memory reachable from that object.

CPython exposes the runtime size of one object through the built-in `sys.getsizeof`
function. For an object $x$, the reported size can be written as

$$
S(x) = \text{bytes occupied by the CPython object representation of } x .
$$

This value is an implementation detail of CPython and can differ between Python
implementations or builds.

## Task

Implement `object_size(x)`:

```python
def object_size(x):
    ...
```

The function takes one Python object and returns its CPython reported size as an
integer number of bytes.

Do not estimate the size from the object's contents. Use the runtime API that
reports the object's size.

## Example

```python
object_size(42)
# returns the same integer as sys.getsizeof(42)

object_size([1, 2, 3])
# returns the same integer as sys.getsizeof([1, 2, 3])
```

## What the gate checks

The gate creates several Python objects and computes the expected values using the
real CPython runtime oracle `sys.getsizeof`. Your function must return exactly the
same integer for every object tested.

The gate checks only the returned size value. It does not accept hardcoded sizes
because the reference value is computed from the active CPython build.
