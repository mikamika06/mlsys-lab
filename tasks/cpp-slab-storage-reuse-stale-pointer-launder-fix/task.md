## Context

When C++ reuses the same storage for multiple distinct objects via
placement-new, [basic.life] says the earlier object's lifetime ENDS the
moment the new one is constructed there — even though the bytes are the
same memory. A pointer captured before that point does not automatically
refer to the new object; using it (or a value read through it beforehand)
after the reuse is not "the new object's current value," it's just stale
data left over from before.

`std::launder(p)` is the standards-mandated way to obtain a pointer that is
genuinely valid for the object now occupying an address — you hand it a
pointer to the storage, it hands back a pointer the compiler is required to
treat as pointing at whatever object is actually alive there right now.

## Task

Fix `slab_reuse_demo()` (declared in `sol.hpp`) in `solve.cpp`:

1. Placement-new a `Slot{5}` into `storage`.
2. Placement-new a **second** `Slot{11}` into the SAME `storage` — this
   ends the first `Slot`'s lifetime and starts a new one there.
3. Return the value held by the storage NOW (`11`) — read it through the
   pointer placement-new #2 itself returned (or through
   `std::launder(...)` applied to a pointer re-derived from the storage
   address). Never through anything captured before step 2.

The shipped implementation reads (and caches into a local) `p1->value`
**before** the reuse, then returns that cached, stale `5` instead of the
current contents.

## Example

The correct run prints:

```
11
```

The broken starter — which reads the OLD object's value before reusing the
storage, then returns that cached value instead of re-reading through a
pointer valid for the new object — prints:

```
5
```

## What the gate checks

The grader compiles `main.cpp` + `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires an **exact match** of the printed value against the
same driver linked with `ref.cpp`. Returning anything read (or cached)
before the second placement-new — instead of the value actually held by the
reused storage afterward — fails the gate.
