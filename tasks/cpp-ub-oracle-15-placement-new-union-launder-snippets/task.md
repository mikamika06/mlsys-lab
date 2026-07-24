## Context

Manual memory lifecycle management in C++ has several rules that, if
violated, cause Undefined Behavior:

1. **Missing destructor:** reusing storage that holds a non-trivially
   destructible object (like `std::string`) without explicitly calling its
   destructor first is UB.
2. **`std::launder`:** if you reuse storage that previously held a `const`
   object, the *original* pointer becomes stale — the compiler is allowed
   to assume a `const` object's value never changes, so it can cache reads
   through that old pointer across the reuse. Accessing the new object
   through that stale pointer without passing it through `std::launder`
   first is UB. Laundering fixes the pointer permanently, not just for one
   access.
3. **Out of lifetime:** accessing storage before an object has been
   constructed in it, or after it has been destroyed, is UB.
4. **Wrong active type:** accessing storage through a type that doesn't
   match the object currently alive there is UB.
5. **Const correctness:** writing to an object that was constructed `const`
   is UB.

## Task

Implement

```cpp
int classify_ub(const Op* ops, int n);
```

which replays `ops[0..n)` (declared in `sol.hpp`) against a single raw
storage location, tracking whether an object is currently alive, its
`type`/`is_const`/`is_trivial`, and whether the original pointer is
currently **stale** (needs laundering):

- `ALLOCATE` — reset: nothing alive, no staleness.
- `PLACEMENT_NEW` — UB if an object is alive and it is **not** trivially
  destructible (its dtor was never called). Otherwise: if *any* object has
  ever occupied this storage and the one just overwritten was `const`, the
  pointer becomes stale (regardless of whether it was destroyed properly
  first — const-ness of the *previous occupant* is what matters). Then the
  new object becomes alive.
- `DTOR` — UB if nothing is alive.
- `ACCESS` — UB if nothing is alive; UB if `op.type` doesn't match the
  alive object's type; UB if `is_write` and the alive object is const; UB
  if the pointer is stale and this access isn't `laundered`. A `laundered`
  access clears staleness going forward.

Return `1` if the trace is UB, `0` if every step is well-defined.

## Example

```
ALLOCATE
PLACEMENT_NEW  int,    const=true,  trivial=true
PLACEMENT_NEW  float,  const=false, trivial=true   <- overwrites a CONST int: pointer now stale
ACCESS         float,  write=false, laundered=false
```
`classify_ub` returns `1`: the original `int` was `const`, so the pointer
became stale on reuse, and this access didn't launder it.

## What the gate checks

The driver runs 11 fixed traces, each isolating one rule (or a combination
— including a proper destructor call that does NOT clear staleness, and
laundering that fixes it permanently rather than just once), and prints
each trace's classification. The grader compiles `solve.cpp` with
`clang++ -O2 -std=c++20`, runs it, and requires

$$
\mathrm{exact\_match} = 1 \iff \text{every one of the 11 classifications matches the reference}
$$

Treating "destructor was called" as sufficient to clear staleness misses
the case where a `const` object is destroyed properly and then a *new*
object is placed in the same storage — the old pointer is still stale for
that new object, because staleness comes from the previous occupant's
const-ness, not from a missing destructor call.
