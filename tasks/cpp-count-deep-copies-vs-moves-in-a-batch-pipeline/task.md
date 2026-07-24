## Context

In C++, whether an operation on an object is a **copy** or a **move**
depends on the *value category* of what's on the right-hand side, not on
what the code "looks like it's doing":

- Passing/returning a **temporary** (a prvalue, like `Buffer()`) selects the
  move overload — there's nothing left to preserve, so the compiler is free
  to steal its guts.
- Passing a **named object** (an lvalue, like a local variable `b`) selects
  the copy overload — `b` might still be used afterward, so its contents
  must be duplicated, not stolen.
- Wrapping a named object in `std::move(x)` doesn't move anything itself; it
  just casts `x` to an rvalue reference, which then selects the move
  overload — you are promising the compiler that `x` won't be used again
  (except to destroy or reassign it).

`std::vector<T>` additionally performs its own moves under the hood: when a
`push_back` needs more capacity than it has, it allocates a bigger block and
relocates every existing element into it — using **move**, not copy,
*provided* `T`'s move constructor is `noexcept` (otherwise it falls back to
copying, to preserve the strong exception guarantee if a mid-relocation copy
throws).

## Task

`Buffer` (declared in `sol.hpp`) has instrumented copy/move special members:
every copy bumps a global `g_copy_count`, every move bumps `g_move_count`,
and its move constructor is `noexcept`. Implement

```cpp
void run_pipeline(const Op* ops, int n, std::vector<Buffer>& vec);
```

which runs `ops[0..n)` against `vec` (starts empty) **in order**, performing
exactly the C++ operation each op names:

| `op.kind` | meaning       | what to write                              |
|-----------|---------------|---------------------------------------------|
| `0`       | `push_temp`   | `vec.push_back(Buffer());`                   |
| `1`       | `push_lvalue` | `Buffer b; vec.push_back(b);`                |
| `2`       | `copy_assign` | `vec[op.dst] = vec[op.src];`                 |
| `3`       | `move_assign` | `vec[op.dst] = std::move(vec[op.src]);`      |

You are not counting anything yourself — `Buffer`'s instrumented special
members and `std::vector`'s own reallocation moves do that. Your only job is
to perform the operation with the *right value category*, so the compiler
picks the copy or move overload the op actually calls for.

## Example

Two `push_temp` in a row on an initially-empty vector: the first allocates
capacity 1 and move-constructs the temporary in (1 move). The second needs
capacity 2, so the vector reallocates — moving the first element into new
storage (1 more move) — and then move-constructs the new temporary in (1
more move). Total: 0 copies, 3 moves.

## What the gate checks

The driver runs two fixed op sequences (mixing all four op kinds, deep
enough to trigger several vector reallocations) and prints
`copies=<n> moves=<n> final_size=<n>` after each. The grader compiles
`solve.cpp` with `clang++ -O2 -std=c++20`, runs it, and requires

$$
\mathrm{exact\_match} = 1 \iff \text{every printed count, in both sequences, matches the reference}
$$

Using `push_back(b)` for `push_temp`, or forgetting `std::move` on
`move_assign`, silently turns a move into a copy — same final vector
contents, but a different (larger) copy count, which the gate catches even
though nothing about the *data* is wrong.
