## Context

When passing objects to functions in C++, the parameter type decides whether
the argument is copied (invoking the copy constructor) or bound by
reference. Since C++17, "guaranteed copy elision" also means passing a
prvalue (a fresh temporary, not a named variable) to a by-value parameter
invokes **no** copy or move constructor at all — the temporary is
materialized directly into the parameter.

The fixed instrumented struct (`sol.hpp`) counts real copy-constructor calls
into a global `g_copy_count`:

```cpp
struct Probe {
    char label;
    int count;
    double data[3];

    Probe();                          // default: fills the fields
    Probe(const Probe& other);        // bumps g_copy_count, then copies
};
```

and three fixed overloads that do nothing with their argument — any copies
that happen are entirely decided by how the *caller* passes it:

```cpp
void process_value(Probe p);
void process_const_ref(const Probe& p);
void process_ref(Probe& p);
```

## Task

Implement `run_scenarios(int out[4])` (declared in `sol.hpp`). Construct one
`Probe obj` (an lvalue), then run these four scenarios **in order**,
resetting `g_copy_count` to `0` immediately before each call and recording
the count observed right after it:

1. `out[0]` — `process_value(obj)`: lvalue passed by value.
2. `out[1]` — `process_const_ref(obj)`: lvalue passed by `const&`.
3. `out[2]` — `process_ref(obj)`: lvalue passed by `&`.
4. `out[3]` — `process_value(Probe{})`: a **fresh prvalue** passed by value —
   construct the temporary directly in the call expression, not in a named
   variable first (naming it first turns it into an lvalue on the next line,
   which would force a real copy).

## Example

For the fixed driver, the correct run prints:

```
1 0 0 0
sizeof(Probe)=32
```

- Scenario 1 copies once (lvalue -> by-value parameter always copies).
- Scenarios 2 and 3 copy zero times (both bind a reference, no new object).
- Scenario 4 copies zero times (guaranteed copy elision on a prvalue).
- `sizeof(Probe) = 32`: `char label` (1 byte) needs 3 bytes of padding
  before the 4-byte-aligned `int count`, then `double data[3]` (24 bytes,
  8-byte aligned) follows directly — 1 + 3 + 4 + 24 = 32, already a multiple
  of the struct's own 8-byte alignment, so no tail padding is added.

A starter that leaves `out` unpopulated (or fills it with placeholder values)
prints `-1 -1 -1 -1`, which cannot match.

## What the gate checks

The grader compiles `main.cpp` + `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires an **exact match** of the four printed copy counts
(plus the fixed `sizeof(Probe)` line) against the same driver linked with
`ref.cpp`. Forgetting to reset the counter between scenarios, calling the
wrong overload, or binding the scenario-4 temporary to a name before passing
it all change a printed count and fail the gate.
