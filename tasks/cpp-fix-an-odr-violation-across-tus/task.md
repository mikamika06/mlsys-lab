## Context

The **One Definition Rule (ODR)** requires an `inline` function or type to be defined identically in every translation unit (TU) that uses it. `inline` exists partly to make this safe: the compiler emits it as a *weak* ("linkonce") symbol in every TU that defines it, and the linker is allowed to keep just one copy and discard the rest, trusting that they're all identical.

If two TUs define `inline int get_size()` with genuinely **different** bodies -- an ODR violation -- the linker still doesn't complain. It just picks one definition (in practice, on this toolchain: whichever object file appears first on the link command line) and silently uses it for *every* call site in the program, including calls made from inside the TU that "lost". `main.cpp` here already defines its own `Config`/`get_size()` this way, unguarded, standing in for a hazard already present in a shared header.

The fix is **internal linkage**: wrap your own conflicting struct and function in an anonymous namespace. That gives them a distinct linker symbol per TU, so there's nothing for the linker to merge.

## Task

Fix `solve.cpp` so it defines its own `struct Config { int x; double y; }` and `get_size()` (returning `sizeof(Config)`) inside an anonymous namespace, and `reportSize()` calls that internally-linked `get_size()`. The shipped version defines them with ordinary external linkage instead, so the real linker merges its `get_size` with `main.cpp`'s same-named one.

## Example

```cpp
namespace {
    struct Config { int x; double y; };
    __attribute__((noinline)) int get_size() { return (int)sizeof(Config); }
}
int reportSize() { return get_size(); }   // correctly reports 16, not main.cpp's 4
```

## What the gate checks

`main.cpp` calls its own `get_size()` (always `4`, `sizeof(Config{int x;})`) and your `reportSize()`, and prints both. With an anonymous namespace, `reportSize()` correctly prints `16` (`sizeof(Config{int x; double y;})`) -- your own file's true, unmerged answer. Without one, the real linker's weak-symbol merge (first object file on the command line wins) makes `reportSize()` silently print `4`, the wrong TU's size. Your printed output is compared byte-for-byte against `ref.cpp`, compiled and linked the same way: `exact_match == 1.0`.
