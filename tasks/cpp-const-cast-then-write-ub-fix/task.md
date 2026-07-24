## Context

In C++, casting away `const`ness with `const_cast` is legal only if the
underlying object was not originally declared `const`. If the object *was*
genuinely declared `const`, writing to it through a `const_cast`
pointer/reference is **Undefined Behavior (UB)** — and real optimizing
compilers exploit exactly this UB, not just in theory.

```cpp
struct Config { char version; double threshold; int flags; };
inline const Config g_cfg = {'A', 0.5, 0};

Config make_config_with_flags(int new_flags) {
    const_cast<Config&>(g_cfg).flags = new_flags;  // UB!
    return g_cfg;
}
```

Because `g_cfg` is genuinely `const` and its initializer is visible to the
compiler, the compiler is free to assume `g_cfg` never changes. At `-O2` it
acts on that assumption: it drops the write to `g_cfg.flags` entirely and
keeps folding every later read of `g_cfg` back to the original initializer.
The function silently returns `flags = 0` no matter what `new_flags` was —
your write vanished, with no crash and no warning at run time.

The fix is to never write through a `const_cast` of a genuinely-const
object. Instead, make a legitimately mutable copy and write to that:

```cpp
Config local = g_cfg;
local.flags = new_flags;
```

## Task

Fix `make_config_with_flags` in `solve.cpp`:

```cpp
Config make_config_with_flags(int new_flags);
```

It must return a `Config` whose `version` and `threshold` match `g_cfg`
(declared in `sol.hpp`) and whose `flags` equals `new_flags` — built via a
genuinely mutable local copy, never via `const_cast` on `g_cfg`.

The fixed driver in `main.cpp` calls your function for several `new_flags`
values and, after each call, also prints `g_cfg` itself, to make explicit
that `g_cfg` must never actually change.

## Example

```
make_config_with_flags(42)
# result: version='A' threshold=0.500 flags=42
# g_cfg :  version='A' threshold=0.500 flags=0   (unchanged)
```

## What the gate checks

The grader compiles `main.cpp` + your `solve.cpp` with real
`clang++ -O2 -std=c++20`, runs it, and compares stdout byte-for-byte against
the reference build (`exact_match == 1.0`). The starter uses the
`const_cast`-then-write pattern: at `-O2` the compiler drops the write and
folds the read, so every fixture prints `flags=0` instead of the requested
value — a real, reproducible miscompile, not a simulated one.
