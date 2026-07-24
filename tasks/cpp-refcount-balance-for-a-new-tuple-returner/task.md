## Context

CPython's C-API is reference-counted: `Py_INCREF`/`Py_DECREF` add and remove
ownership of a `PyObject*`, and getting the count wrong means either a
leak (an object outlives its last real owner) or a use-after-free/double-free
(an object is freed while something still points at it). This task models
that discipline with a small, self-contained C++ type — `PyObj` — so the
lesson can be built and graded without CPython headers, but the rule is
exactly CPython's:

`PyTuple_SetItem(tuple, i, item)` **steals** a reference to `item` — it does
NOT call `Py_INCREF` on your behalf. Two valid ways to hand it an item:

1. **Fresh reference**: `Py_INCREF(item); PyTuple_SetItem(tuple, i, item);`
   — you created a new reference just for the tuple to steal, so your own
   reference to `item` is untouched afterward. Net delta on `item`: **+1**.
2. **Stolen reference**: `PyTuple_SetItem(tuple, i, item);` (no incref) —
   the tuple now owns *your* reference. You must not use it independently
   afterward. Net delta on `item`: **0**.

Either way, the tuple itself comes back as a **new reference** with
`refcount == 1` that the caller now owns.

## Task

Implement

```cpp
PyTuple* make_tuple(PyObj** items, int n, bool increment_input_refs);
```

- If `increment_input_refs` is `true`, call `incref()` on each of
  `items[0..n)` **before** handing it to the tuple.
- If `false`, hand each item to the tuple directly, with no `incref()` call.
- Either way: allocate a new `PyTuple`, set its `refcount` to `1`, copy
  `items[0..n)` into its `items[]` array in order, and return it.

## Example

```cpp
PyObj a{1, 0};                       // caller already owns 1 reference to a
PyObj* items[] = {&a};
PyTuple* t = make_tuple(items, 1, /*increment_input_refs=*/true);
// a.refcount == 2   (the fresh reference the tuple now owns)
// t->refcount == 1  (the caller's new reference to the tuple)
```

## What the gate checks

The driver runs both scenarios (`increment_input_refs` true, then false) on
three items, printing each item's refcount delta, the resulting tuple's
refcount/length/contents, and the total `incref`/`decref` call counts. The
grader compiles `solve.cpp` with `clang++ -O2 -std=c++20`, runs it, and
requires

$$
\mathrm{exact\_match} = 1 \iff \text{every printed delta, tuple field, and call count matches the reference}
$$

Always increfing the items (regardless of the flag) passes the
`increment_input_refs=true` case but leaves the "stolen reference" case with
deltas of `+1` where the reference is expected to reflect a real steal
(`0`) — a real CPython extension with this bug leaks one reference to every
item passed with `increment_input_refs=false`.
