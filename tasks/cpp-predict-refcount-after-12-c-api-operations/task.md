## Context

In CPython every `PyObject` carries a reference count (`ob_refcnt`) that
starts at 1 when the object is created. `Py_INCREF`/`Py_DECREF` increment
and decrement it; `Py_NewRef` hands out a new (owned) reference, which also
increments it; a *borrowed* reference (e.g. from `PyList_GetItem`) does
**not** change the count at all. We encode a sequence of such operations as
strings:

* `"New"` – a new owned reference to the object (increments count by 1).
* `"Incref"` – `Py_INCREF` (increments by 1).
* `"Decref"` – `Py_DECREF` (decrements by 1).
* `"Borrow"` – a borrowed reference (no change to the count).

So the refcount after a sequence of operations, starting from a fresh
object (initial refcount = 1), is

$$
\text{rc} = 1 + (\#\text{Incref} + \#\text{New}) - \#\text{Decref}.
$$

## Task

Implement

```cpp
void predict_refcounts(int out[12]);
```

Write your predicted final refcount for each of the 12 fixed operation
sequences below into `out[0..12)` (`out[i]` is sequence `i+1`), starting
from refcount 1. You don't need to handle deallocation — the count may
reach zero without error.

1. `["New"]`
2. `["Incref", "Decref"]`
3. `["Incref", "Incref", "Decref"]`
4. `["Borrow", "Incref"]`
5. `["Incref", "Borrow", "Decref", "Decref"]`
6. `["New", "Incref", "Borrow", "Decref", "Decref"]`
7. `["New", "New"]`
8. `["Borrow", "Borrow"]`
9. `["Incref", "Incref", "Incref", "Decref", "Decref", "Decref"]`
10. `["Decref"]`
11. `["Incref", "Decref", "Incref", "Decref", "Incref"]`
12. `["New", "Borrow", "Incref", "Decref", "New", "Decref"]`

## Example

Sequence 1, `["New"]`: start at 1, `New` → 2.
Sequence 2, `["Incref", "Decref"]`: start at 1, `Incref` → 2, `Decref` → 1.

## What the gate checks

`main.cpp` builds a real intrusively-refcounted `Obj` (an `ob_refcnt`
field) with real `Py_INCREF_`/`Py_DECREF_`/`Py_NewRef_`/`Py_Borrow_`
functions, actually runs each of the 12 sequences against a fresh real
object, and reads back the real resulting `ob_refcnt` as ground truth —
never a hardcoded table. It compares your 12 predictions against that,
prints a line per sequence plus the match count, and the grader requires
your `.cpp`'s output (compiled with the real local `clang++`) to match the
reference's exactly ($\mathrm{exact\_match}=1.0$).
