## Context

When writing a CPython C-API extension, understanding the reference-counting model is critical.

- `PyList_GetItem` returns a **borrowed reference**. You do not own it, so you should not call `Py_DECREF` on it unless you first called `Py_INCREF` yourself.
- `PyTuple_SetItem` **steals a reference**. It assumes ownership of whatever reference you pass in.

If you pass a borrowed reference straight to a function that steals it, the receiving object (the tuple) takes ownership of a reference you never actually owned. When something later drops that reference, the refcount goes to zero while the original owner (the list) still thinks it holds a valid one — a double-free / use-after-free.

`sol.hpp` gives the base CPython object layout under LP64 (`Py_ssize_t` is `long`):

```cpp
struct PyObject {
    long ob_refcnt;
    void* ob_type;
};
```

## Task

Implement `void* process_list_to_tuple(void* list)`, plus `Py_INCREF`/`Py_DECREF` themselves (direct `obj->ob_refcnt` field access — this task does not model deallocation at refcount `0`).

`process_list_to_tuple` must build a new tuple containing every item of `list`, using the mock C-API declared in `sol.hpp` (`PyList_Size`, `PyList_GetItem`, `PyTuple_New`, `PyTuple_SetItem`), while respecting CPython's ownership rules: `PyList_GetItem` hands you a *borrowed* reference, and `PyTuple_SetItem` *steals* whatever reference you give it. To hand the tuple a reference it's allowed to steal, you must `Py_INCREF` the borrowed item yourself first.

## Example

Broken: `item = PyList_GetItem(list, i)` (refcnt stays `1`, borrowed) → `PyTuple_SetItem(tup, i, item)` (tuple now thinks it owns a reference) → `Py_DECREF(item)` (refcnt drops to `0` — destroyed while the list and tuple both still point at it).

Correct: `item = PyList_GetItem(list, i)` → `Py_INCREF(item)` (refcnt `1` → `2`, now you genuinely own one) → `PyTuple_SetItem(tup, i, item)` (steals the reference you just took; refcount stays `2`, correctly shared between list and tuple).

## What the gate checks

`main.cpp` builds three real `PyObject`s with `ob_refcnt == 1`, calls `process_list_to_tuple`, and prints the returned tuple handle, whether each tuple slot holds the right item, and every item's final `ob_refcnt`. The candidate's full stdout is compared byte-for-byte (`exact_match = 1.0`) against the reference's stdout, whose correct implementation leaves every `ob_refcnt` at exactly `2`. Over-decrefing a borrowed reference drives the count to `0` instead — the exact bug this task is about, and one that would corrupt real interpreter memory.
