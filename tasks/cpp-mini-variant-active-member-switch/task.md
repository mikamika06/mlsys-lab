## Context

A tagged union (or `std::variant`) holds exactly one active alternative at a
time. Switching the active member means: destroy the currently-active
object first (`~A()`), THEN placement-`new` the new one into the same
storage — never the other order, and never skip the destroy just because
the new type happens to equal the old one.

The fixed contract (`sol.hpp`) instruments two alternative types so every
real constructor/destructor call is directly observable:

```cpp
struct TypeA { int x; double y;  /* ctor logs "ctor_TypeA", dtor logs "dtor_TypeA" */ };
struct TypeB { char c; long l;   /* ctor logs "ctor_TypeB", dtor logs "dtor_TypeB" */ };

struct MiniVariant {
    union Storage { TypeA a; TypeB b; /* does nothing on its own */ };
    int active = 0;   // 0 = none, 1 = TypeA live, 2 = TypeB live
    Storage storage;
};
```

`Storage`'s own constructor/destructor are empty — nothing about the union
itself manages the active member's lifetime. That is entirely up to the
five functions you implement.

## Task

Implement, in `solve.cpp`:

- `variant_set_a(v)` / `variant_set_b(v)` — if a member is currently active,
  destroy it (explicit destructor call) **first**, THEN placement-`new` the
  requested type into `v.storage`, then set `v.active`.
- `variant_get_a(v)` / `variant_get_b(v)` — log `"access_TypeA"` /
  `"access_TypeB"` if that type is the active member, otherwise
  `"invalid_access"`. Never touch the storage's lifetime.
- `variant_destroy(v)` — destroy the active member if any, and reset
  `v.active` to `0`.

## Example

The fixed driver replays four operation sequences against four independent
`MiniVariant`s (`set A, get A, set B, get B, destroy`; `set A, get B (wrong
type), set B, get B`; `set A, set A again (same type — still destroy+
reconstruct), get A, set B`; `get A (nothing active), set A, get A, set B`),
each followed by a trailing `variant_destroy`. The correct run prints:

```
ctor_TypeA
access_TypeA
dtor_TypeA
ctor_TypeB
access_TypeB
dtor_TypeB
ctor_TypeA
invalid_access
dtor_TypeA
ctor_TypeB
access_TypeB
dtor_TypeB
ctor_TypeA
dtor_TypeA
ctor_TypeA
access_TypeA
dtor_TypeA
ctor_TypeB
dtor_TypeB
invalid_access
ctor_TypeA
access_TypeA
dtor_TypeA
ctor_TypeB
dtor_TypeB
sizeof(MiniVariant)=24
```

Notice case 3's `set A` while `A` is already active still logs
`ctor_TypeA` / `dtor_TypeA` — a real `std::variant` doesn't special-case
"same type," it always destroys and reconstructs. A starter with empty
function bodies logs nothing at all except the fixed `sizeof` line.

## What the gate checks

The grader compiles `main.cpp` + `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires an **exact match** of the full printed event log
against the same driver linked with `ref.cpp`. Constructing before
destroying, skipping the destroy on a same-type `set`, or checking the
wrong `active` value on `get` all reorder or drop events and fail the gate.
