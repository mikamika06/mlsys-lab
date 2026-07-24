## Context

Returning a reference (`const&`) to a local variable or temporary results in
a dangling reference: the referand is destroyed when the function returns,
so reading through the reference afterward is undefined behavior — usually
manifesting as garbage or zeroed memory once something else reuses that
stack space.

The contract (`sol.hpp`) is a fixed record and a function that must return it
**by value**:

```cpp
struct Result { int id; float val; };
Result get_result(int id, float val);
```

The shipped (broken) implementation still matches that by-value signature on
the outside, but internally routes through a helper that returns a `const&`
bound directly to a temporary:

```cpp
static const Result& make_dangling(int id, float val) {
    return Result{id, val};   // temporary's lifetime ends when THIS returns
}
```

By the time `get_result` reads through that reference, the temporary is
gone — the classic "reference to local/temporary" bug, just one call frame
removed from the textbook one-liner.

## Task

Fix `get_result` in `solve.cpp` so it returns the constructed `Result` **by
value**, with no reference — direct or indirect — outliving the temporary it
was bound to. `sol.hpp`'s declared signature already returns `Result` by
value; the bug is entirely in how the current body gets there.

## Example

For the fixed driver's three cases the correct run prints:

```
1 3.140000
2 -2.500000
42 10.000000
```

The broken starter still gets `id` right (it happens to survive in this
build) but `val` reads back as `0.000000` in all three cases — the float
field was clobbered by the scratch write in `make_dangling`'s now-dead stack
frame before `get_result` ever read it.

## What the gate checks

The grader compiles `main.cpp` + `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires `max_abs_err <= 1e-6` against the same driver linked
with `ref.cpp`. A fix that still funnels the value through any reference to
a temporary — instead of constructing and returning a real `Result` by value
all the way out — reintroduces the dangling read and fails the gate.
