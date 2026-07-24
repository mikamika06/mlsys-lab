## Context

Every Python object begins with a common object header. On a 64-bit CPython build, the base header for ordinary objects contains a reference count field and a type pointer:

$$
\text{header\_bytes} = 8 + 8 = 16 .
$$

The total memory reported by CPython for an object is given by `sys.getsizeof`. The remaining bytes can be viewed as the object's payload storage:

$$
\text{payload\_bytes} = \text{getsizeof}(x) - \text{header\_bytes}.
$$

Different builtin objects store different amounts of data after the universal header. For example, an integer stores digit data, a tuple stores references to its elements, and a list stores bookkeeping information plus a pointer array.

This task uses the CPython runtime itself as the oracle. The answer depends on the interpreter's object layout, so the grader measures the current pinned CPython 64-bit environment rather than using manually written sizes.

## Task

Implement `header_byte_map(objects)`:

```python
def header_byte_map(objects):
    ...
```

The function receives a list of builtin Python objects and returns a list of rows. Each row must contain three integers:

```python
[
    total_size,
    header_size,
    payload_size
]
```

where:

- `total_size` is `sys.getsizeof(obj)`.
- `header_size` is the fixed CPython object header size of $16$ bytes.
- `payload_size` is `total_size - 16`.

The function should preserve the input order and return only Python integers.

## Example

```python
items = [1, 3.5, "abc", (1, 2), True, None]

result = header_byte_map(items)

# Example shape:
# [
#   [28, 16, 12],
#   [24, 16, 8],
#   [52, 16, 36],
#   [56, 16, 40],
#   [28, 16, 12],
#   [16, 16, 0]
# ]
```

The exact total sizes are determined by the CPython build used by the grader.

## What the gate checks

The gate creates a fixture list containing builtin objects such as integers, floats, strings, tuples, lists, bytes, booleans, and `None`.

The grader computes the expected table directly with CPython's `sys.getsizeof` and compares the returned integer table exactly. A solution that assumes object sizes or uses a hardcoded table will fail when the measured objects differ.
