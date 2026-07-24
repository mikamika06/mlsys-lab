## Context

When writing Python C extensions, managing `PyObject` reference counts is
critical. C-API calls fall into two categories: **borrowed** references
(e.g. `PyTuple_GetItem` — you don't own it, don't decref it) and **new**
references (e.g. `PyObject_GetAttrString` — you own it and **must**
`Py_DECREF` it when done). A classic bug: acquiring a new reference, then
returning early on an error path without releasing it first — a leak.

The fixed contract (`sol.hpp`) models this with a simplified record and its
ref-counting ops:

```cpp
struct MyPyObject { long ob_refcnt; void* ob_type; long value; };
void obj_incref(MyPyObject* obj);   // like Py_INCREF
void obj_decref(MyPyObject* obj);   // like Py_DECREF
```

## Task

Fix `process_items(MyPyObject* objs, int n)` in `solve.cpp`. For each
object, in order: acquire a new/owned reference (`obj_incref`, as if calling
`PyObject_GetAttrString`), then check `value`:

- If `value < 0` — an error — release the reference you just acquired
  (`obj_decref`) **before** returning `-1`. Do not touch any later object.
- Otherwise release the reference (`obj_decref`) and continue to the next
  object.

Return `0` after every object has been processed without error.

The invariant: after the call, every object's `ob_refcnt` must be back to
exactly what it was before the call — no net change, on either the success
path or the error path.

## Example

The fixed driver builds 5 objects (`ob_refcnt = 1` each) with values
`10, 20, 30, -5, 40`. Processing stops at the 4th object (`value = -5`). The
correct run prints:

```
rc=-1
1 1 1 1 1
```

The shipped starter acquires the reference on that 4th object but never
releases it before returning, so it still reports `rc=-1` but leaves that
object's `ob_refcnt` at `2`:

```
rc=-1
1 1 1 2 1
```

## What the gate checks

The grader compiles `main.cpp` + `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires an **exact match** of the printed return code and all
five final `ob_refcnt` values against the same driver linked with `ref.cpp`.
Any missing `obj_decref` on the error path — or an extra one that
under-releases a later object — changes a refcount and fails the gate.
