## Context

C++ compilers use **name mangling** to encode function signatures into unique linker symbols, so overloads, namespaces, and classes can all coexist in one flat symbol table. The Itanium C++ ABI specifies the mangling scheme used by clang and gcc on macOS and Linux, and every C++ toolchain ships a real demangler for it -- `c++filt` on the command line, and `abi::__cxa_demangle()` (declared in `<cxxabi.h>`) as a callable function linked into every C++ binary.

In this ABI, mangled names start with `_Z`.
- Length-prefixed identifiers: `3foo` means `foo`.
- `N ... E` wraps a nested name (namespace/class member): `N1S1fE` means `S::f`.
- `K` right after `N` marks a `const` member function.
- `v` means "no parameters"; `i` is `int`, `d` is `double`.
- `P` means pointer, `R` means reference, and `K` right after `P`/`R` means the pointee/referent is `const`.

## Task

Implement `demangleOne(mangled)` in `solve.cpp`: a hand-written parser for the restricted grammar spelled out in `sol.hpp`, covering exactly the forms `main.cpp` exercises (free functions and possibly-nested, possibly-const member functions, with `int`/`double` parameters optionally behind one level of pointer or reference, optionally const).

Match the real demangler's exact formatting: `const` on a pointer/reference target goes **after** the base type (`"double const*"`, not `"const double*"`), no return type is ever printed, and a `const` member function gets `" const"` appended at the very end.

## Example

```cpp
demangleOne("_Z3fooii");        // -> "foo(int, int)"
demangleOne("_ZN1S1fEv");       // -> "S::f()"
demangleOne("_Z4funcPKd");      // -> "func(double const*)"
demangleOne("_ZNK5Outer1fEPi"); // -> "Outer::f(int*) const"
```

## What the gate checks

`ref.cpp` doesn't hand-code any expected strings -- it forwards straight to `abi::__cxa_demangle()`, the actual production Itanium demangler from libc++abi, the same code path a real linker or debugger uses. `main.cpp` runs both your parser and the reference through the same 12 mangled names and prints `"<mangled> -> <demangled>"` for each. The full printed output is compared byte-for-byte against the reference build: `exact_match == 1.0`. Getting the `const`-placement convention backwards, dropping the `::` between nested names, or printing a return type will all produce mismatched text somewhere in the 12 lines.
