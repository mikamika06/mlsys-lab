## Context

Every CPython object starts with a universal header called `PyObject`. It stores a reference count and a pointer to the object's type. Variable-sized objects extend this layout with a `PyVarObject` field containing the number of items.

For a 64-bit CPython build, this task models the header as three little-endian unsigned 64-bit fields:

$$
\mathrm{header} = \mathrm{pack}_{<Q}(ob\_refcnt) \; || \; \mathrm{pack}_{<Q}(type\_id) \; || \; \mathrm{pack}_{<Q}(ob\_size)
$$

where $||$ denotes byte concatenation. For fixed-size objects, $ob\_size$ is zero. For variable-sized objects, $ob\_size$ is the object's item count.

The type pointer is represented by a supplied integer table rather than the process address. The table maps Python types to stable ids for the fixture.

## Task

Implement `pack_pyobject_header(obj, type_ids, is_var_object)`.

The function must return a `bytes` object containing the modeled header.

Arguments:

- `obj`: a Python object whose CPython header is inspected.
- `type_ids`: a dictionary mapping Python types to integer ids.
- `is_var_object`: a boolean. If true, include the object's variable-size item count.

The packed fields must be:

1. The object's current reference count.
2. `type_ids[type(obj)]`.
3. `len(obj)` when `is_var_object` is true, otherwise zero.

Use little-endian unsigned 64-bit packing.

## Example

```python
obj = [10, 20, 30]
type_ids = {list: 7}

data = pack_pyobject_header(obj, type_ids, True)

# data contains:
# refcount as <Q
# 7 as <Q
# 3 as <Q
```

## What the gate checks

The gate compares the returned bytes against a CPython memory-layout oracle. The oracle reads the live `PyObject` and `PyVarObject` header fields from the interpreter using the actual object address.

The metric is byte equality:

$$
\mathrm{score} =
\frac{\text{matching bytes}}{\text{total bytes}}
$$

The required score is $1.0$. Any incorrect packing order, integer size, byte order, type id handling, or variable-object size handling fails.
