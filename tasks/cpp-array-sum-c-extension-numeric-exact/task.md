## Context

A CPython C-API extension function does not receive plain doubles: it
receives `PyObject*` handles with a manual reference count, and it must
follow strict ownership rules or it corrupts the interpreter's memory:

- A **borrowed reference** (an argument the caller passes you) is yours to
  *read* during the call, but you must not change its refcount unless you
  are keeping a handle to it beyond the call's lifetime. Extra `Py_INCREF`
  without a matching `Py_DECREF` leaks the object forever; an unmatched
  `Py_DECREF` can drop someone else's live object to refcount 0 and free
  memory that is still in use elsewhere.
- A **new reference** (an object you allocate and return) starts life
  owned by you, with refcount `1`, and that single reference passes to
  your caller when you return it.

This task models that discipline with a minimal `PyObj` stand-in (a
`double* data`, `int n`, `long refcount`) and the two primitives every
extension calls, `Py_INCREF` / `Py_DECREF`, where `Py_DECREF` genuinely
frees the payload once the refcount reaches zero -- mirroring CPython's real
deallocation.

## Task

Implement, in `solve.cpp`,

```cpp
PyObj* array_sum_ext(PyObj* buffer);
```

`buffer` is a **borrowed** reference holding `buffer->n` doubles at
`buffer->data`.

1. Compute `sum = buffer->data[0] + ... + buffer->data[n-1]`.
2. Allocate and return a brand-new `PyObj*` that owns a freshly allocated
   one-element `double[]` holding `sum`, with `refcount == 1` -- a **new**
   reference the caller now owns.
3. Do **not** change `buffer`'s refcount. You only borrowed it and never
   took ownership, so its refcount on return must equal its refcount on
   entry -- no leak, no accidental free of the caller's buffer.

## Example

For the fixed 8-element buffer
`{1.5, -2.25, 3.0, 4.75, -0.5, 2.0, 0.25, -1.0}` (sum `= 7.75`), owned by
the caller with `refcount = 1` on entry, a correct implementation leaves the
driver printing:

```
sum=7.7500000000 result_refcount=1 buffer_refcount=1 buffer_alive=1
```

`buffer_refcount` and `buffer_alive` unchanged from their entry values
(`1` and `1`) prove you never touched the borrowed reference's lifetime;
`result_refcount=1` proves the object you handed back is a clean new
reference.

## What the gate checks

The fixed driver (`main.cpp`) builds the buffer object, calls
`array_sum_ext`, and prints the sum plus the full refcount/aliveness
picture in one line. The gate is an exact string match
(`exact_match == 1.0`) against the reference's printed line: a wrong sum,
a missing/null result, a result with the wrong refcount, or any change to
the borrowed buffer's refcount or liveness all produce a different line
and fail the gate -- not just the numeric value has to be right, the whole
C-API contract does.
