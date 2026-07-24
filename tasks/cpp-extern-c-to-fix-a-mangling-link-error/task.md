## Context

C++ compilers encode extra information into exported symbol names — this is
called **name mangling**, and it's what lets overloaded functions with the
same source-level name coexist as distinct linker symbols. A C-style
consumer doesn't know anything about mangling: it links against the plain,
unmangled name.

A link error occurs when the consumer requests symbol $s$ but the object
file exports a different, mangled symbol $m(s)$. The `extern "C"` language
linkage specification tells the compiler to export a function under its
plain, unmangled name — keeping it callable from a C-style consumer while
its implementation is still written in C++.

## Task

`solve.cpp` defines `add(int, int)` — a bridge that should call the C++
function `cpp_add` (declared in `sol.hpp`, defined in `main.cpp`) and return
its result:

```cpp
int add(int a, int b) {
    return cpp_add(a, b);
}
```

This compiles fine on its own, but `main.cpp` (playing the C-style consumer)
forward-declares `add` with C linkage and calls it — and the object file
`solve.cpp` produces doesn't export the symbol the linker is looking for.
Fix the language linkage of `add` so the whole program actually links.

## Example

```cpp
// what main.cpp expects to link against:
extern "C" int add(int a, int b);
```

A correct bridge gives `add` C linkage:

```cpp
extern "C" int add(int a, int b) {
    return cpp_add(a, b);
}
```

## What the gate checks

The grader compiles `main.cpp` together with your `.cpp` using the real
local `clang++` and links them into one binary. Without `extern "C"`, the
C++ compiler mangles `add`'s exported name (something like `_Z3addii` under
the Itanium ABI actually used here) instead of the plain `add` the consumer
in `main.cpp` asks the linker for — so the **link itself fails**, which
fails the gate exactly like a wrong runtime answer would. With the correct
language linkage, the link succeeds and the program prints the sum
($\mathrm{exact\_match}=1.0$ against the reference's output).
