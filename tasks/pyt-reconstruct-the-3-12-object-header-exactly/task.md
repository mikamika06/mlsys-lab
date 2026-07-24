## Context

Every CPython object begins with a common object header. In CPython 3.12 on a 64-bit build, the header consists of two machine-sized fields:

$$
\text{PyObject} = (\text{ob\_refcnt}, \text{ob\_type})
$$

The reference count is a signed machine word and the type field stores a pointer to the object's type. The byte representation therefore contains:

$$
\text{header bytes} =
\text{pack}_{native}(\text{ob\_refcnt}) \Vert \text{pack}_{native}(\text{ob\_type})
$$

where $\Vert$ denotes byte concatenation.

CPython 3.12 also supports immortal objects. Immortal objects keep a special large reference count sentinel instead of changing the header layout. A reconstruction must emit the reference count value that CPython exposes for the object, whether it is an ordinary mortal object or an immortal object.

## Task

Implement `pack_object_header(obj)`:

```python
def pack_object_header(obj) -> bytes:
    ...
```

Return exactly the first two fields of the CPython 3.12 object header as native-endian bytes.

The returned bytes must contain:

- the true reference count value for `obj`;
- the pointer value of `type(obj)`.

Use only the public Python runtime and standard library facilities. The result must be exactly 16 bytes on the grader's 64-bit CPython build.

## Example

```python
x = []
header = pack_object_header(x)

# header contains:
# native bytes of sys.getrefcount(x) - 1
# followed by native bytes of id(type(x))
```

## What the gate checks

The gate constructs mortal objects and objects with CPython immortal reference counts. It computes the expected bytes from the running CPython interpreter using the actual reference count and type pointer values, then compares the returned byte buffer.

The `byte_exact_fraction` score must equal $1.0$. Any difference in one byte fails the task.
