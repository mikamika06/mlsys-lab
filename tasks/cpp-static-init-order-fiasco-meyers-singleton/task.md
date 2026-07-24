## Context

The C++ standard guarantees that namespace-scope globals within ONE
translation unit are dynamically initialized in the order they're declared
— but it says NOTHING about the relative order between globals defined in
DIFFERENT translation units. If one TU's global's constructor depends on
another TU's global already being initialized, you're gambling on link
order. This is the "static initialization order fiasco."

The fixed driver (`main.cpp`) is exactly this gamble, made concrete: its own
global's initializer calls `get_b_value()` while `main.cpp` itself is still
being statically initialized —

```cpp
int g_a_derived = get_b_value() + 100;   // in main.cpp, runs during main.cpp's own static init
```

— and `main.cpp` is compiled/linked FIRST (`clang++ ... main.cpp <src>`),
so its dynamic initializers run before the other translation unit's.

## Task

Implement `get_b_value()` (declared in `sol.hpp`) in `solve.cpp` so it
returns `42`, safely, no matter which TU's static initializers happen to
run first. The fix is the **Meyers singleton**: wrap the value in a
function-LOCAL `static`, not a namespace-scope global:

```cpp
int get_b_value() {
    static int value = /* computed once, e.g. 42 */;
    return value;
}
```

A function-local `static` is guaranteed by the standard to be constructed
the first time control passes through its declaration — wherever that first
call comes from, even from another translation unit's dynamic initializer
running before the rest of THIS translation unit has initialized. A plain
namespace-scope global backing `get_b_value()` has no such guarantee: it's
only safe to read once ITS OWN TU's dynamic initializers have run, and
nothing forces that to happen before `main.cpp`'s does.

## Example

The correct run (Meyers singleton) prints:

```
g_a_derived=142
```

The broken starter — `get_b_value()` returning a plain namespace-scope
global that is itself dynamically initialized — prints:

```
g_a_derived=100
```

`100` is `0 + 100`: `main.cpp`'s initializer called `get_b_value()` before
the other TU's own dynamic initializers had run, so the global it reads was
still sitting at its static (zero) pre-construction value.

## What the gate checks

The grader compiles `main.cpp` + `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires an **exact match** of the printed value against the
same driver linked with `ref.cpp`. Any implementation of `get_b_value()`
that reads a namespace-scope global instead of a function-local static —
even if it "looks" equivalent — reproduces the fiasco under this exact,
fixed link order and fails the gate.
