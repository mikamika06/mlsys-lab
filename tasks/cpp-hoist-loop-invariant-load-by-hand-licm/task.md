## Context

Performance can often be improved by manually hoisting loop-invariant loads
out of a loop (Loop Invariant Code Motion, or LICM). When a field from a
struct is read inside a loop but never modified, the compiler may not
always be able to prove it's safe to hoist (e.g. because of potential
pointer aliasing between `pts` and `state`). Manually loading the value into
a local variable before the loop guarantees fewer memory accesses,
regardless of what the optimizer can prove.

```c
struct Point { double x, y, z; };
struct State { int active; struct Point center; };  // center at offset 8 (LP64)
```

`mem_read_double` / `mem_write_double` (declared in `sol.hpp`, defined in
`main.cpp`) are instrumented: every call to `mem_read_double` increments a
counter, `g_load_count`, so the number of real loads your code performs is
directly observable.

## Task

Implement `apply_shift` in `solve.cpp`:

```cpp
void apply_shift(Point* pts, int n, const State* state);
```

Add `state->center.x/y/z` to every `pts[i].x/y/z`. Access every double
through `mem_read_double` / `mem_write_double`, never through a raw `->` or
`.` read. `state->center` never changes across the loop — hoist its three
reads out: read `center.x`, `center.y`, `center.z` into locals exactly once,
before the loop, and reuse them for every point.

The fixed driver in `main.cpp` resets `g_load_count`, calls `apply_shift`,
and prints the resulting load count followed by every shifted point.

## Example

Unhoisted (too many loads — do not do this):

```cpp
for (int i = 0; i < n; i++) {
    double cx = mem_read_double(&state->center.x);   // re-read every iteration!
    double px = mem_read_double(&pts[i].x);
    mem_write_double(&pts[i].x, px + cx);
    // ... same problem for y and z
}
```

Hoisted (read `center.x/y/z` once, before the loop, then just use `cx, cy,
cz` inside it).

## What the gate checks

The grader compiles `main.cpp` + your `solve.cpp` with real
`clang++ -O2 -std=c++20`, runs it, and compares stdout byte-for-byte against
the reference build (`exact_match == 1.0`) — this includes the printed load
count, not just the shifted point values. The starter re-reads
`state->center.x/y/z` inside the loop: it produces the numerically correct
shifted points, but with $6n$ loads instead of the reference's $3 + 3n$, so
the printed load count differs and the whole comparison fails.
