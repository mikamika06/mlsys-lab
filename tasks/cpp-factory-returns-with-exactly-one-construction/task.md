## Context

In older C++, returning a large object by value could construct a temporary
on the callee's stack and then copy or move it into the caller's storage.
C++17 introduced **guaranteed copy elision**: when a function returns a
*prvalue* (a plain, unnamed value-producing expression like `Probe(a, b)` or
`T(...)`), the compiler is **required** by the standard -- not merely
permitted as an optimization -- to construct that object directly in the
storage the caller provided. No temporary, no copy, no move: exactly one
construction happens, full stop, at every optimization level including `-O0`.

This is different from **NRVO** (Named RVO), which applies when a function
returns a *named local variable* (`T local(...); return local;`). NRVO is
only an optional "as-if" optimization the compiler is allowed, but never
required, to perform.

## Task

Implement

```cpp
Probe make_probe(int tag, double payload);
```

`Probe` (declared in `sol.hpp`) is instrumented: its direct constructor,
copy constructor, and move constructor each bump their own global counter.
Return the freshly built `Probe(tag, payload)` in a way that relies on
**guaranteed** elision, so that across the whole call exactly one
constructor runs in total: the direct one.

## Example

```cpp
Probe make_probe(int tag, double payload) {
    return Probe(tag, payload);   // prvalue: guaranteed elision
}
// g_direct_count == 1, g_copy_count == 0, g_move_count == 0
```

Contrast with a version that looks similar but is **not** guaranteed-elided:

```cpp
Probe make_probe(int tag, double payload) {
    Probe local(tag, payload);    // 1 direct construction
    return Probe(local);          // a REAL copy from an existing named
                                   // object -- copying `local`'s value is
                                   // unavoidable work, so this is a genuine
                                   // extra construction no matter how the
                                   // compiler optimizes the return itself.
}
// g_direct_count == 1, g_copy_count == 1  -> total 2, not 1
```

## What the gate checks

The driver calls `make_probe` for four fixed `(tag, payload)` cases and
prints the resulting fields plus `direct`, `copy`, `move`, and their sum
`total`, all read from `Probe`'s own real constructors -- never hand-counted.
The grader compiles `solve.cpp` with `clang++ -O2 -std=c++20`, runs it, and
requires

$$ \mathrm{exact\_match} = 1.0 $$

against the reference, which prints `total=1` for every case. Introducing
any named intermediate object that gets copied or moved back out -- even one
line, even if the *outer* return statement could itself be optimized away --
pushes `total` to 2 or more and the printed trace stops matching.
