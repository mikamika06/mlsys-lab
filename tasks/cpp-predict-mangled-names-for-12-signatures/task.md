## Context

C++ compilers emit **mangled names** that encode a function's name and
parameter types into its linker symbol, so that overloads get distinct
symbols. The Itanium C++ ABI (used by Clang and GCC on macOS/Linux)
defines a deterministic scheme: a mangled name begins with `_Z`, followed
by the function name encoded as *length* + *literal letters*, followed by
the parameter-type codes — `int`→`i`, `char`→`c`, `double`→`d`,
`void`→`v`, `long`→`l`, `float`→`f`, `bool`→`b`, and a pointer `T*` as
`P` + the code of `T`. A function taking no parameters is mangled as if
it had a single `void` parameter. The return type is **never** part of
the mangled name.

(You can check this yourself: `nm` on an object file compiled from `int
bar(int) { return 0; }` shows the linker symbol `__Z3bari` on macOS — the
`_Z3bari` mangled name with Mach-O's extra leading underscore.)

## Task

Implement, in `solve.cpp`,

```cpp
std::vector<std::string> mangle_signatures(const std::vector<std::string>& sigs);
```

Each element of `sigs` has the exact format
`"return_type function_name(param1_type, param2_type, ...)"`, with
exactly one space between the return type and the function name, and
`", "` between parameter types. Return the corresponding Itanium ABI
mangled name for each, in the same order:

1. Split off the function name (everything between the first space and
   the `(`).
2. Parse the comma-separated parameter type list between the parentheses
   (an empty list, or the literal `void`, both mean "one `void`
   parameter").
3. Emit `"_Z"` + the name's length (decimal) + the name itself, followed
   by each parameter's type code in order (pointers recursively as `P` +
   the pointee's code).

## Example

`"void foo()"` → `"_Z3foov"`. `"int bar(int, double)"` → `"_Z3barid"`.
`"void quux(int*)"` → `"_Z4quuxPi"`.

## What the gate checks

The fixed driver (`main.cpp`) calls `mangle_signatures` with 12 fixed
signatures covering every primitive type, pointers to them, and the
empty-parameter-list case, and prints every resulting mangled name. The
gate is an exact string match (`exact_match == 1.0`) against the
reference's printed output — the reference implements the same parsing
and encoding rules rather than hardcoding the 12 answers, so every
signature must be mangled correctly, not memorized.
