## Context

In C++, every distinct set of template arguments produces a genuinely separate function instantiation, with its own linker symbol and its own generated machine code. A function template

```cpp
template<class T> void f(T) { /* ... */ }
```

called with `int`, `float`, and `double` yields **three** distinct instantiations `f<int>`, `f<float>`, `f<double>` -- even though they're all "just numbers". Each one bloats the binary independently. The standard fix, when several call sites are conceptually interchangeable, is to convert to a single common type (or a shared base / type-erased interface) *before* the templated call, so the compiler only ever has to instantiate it once.

## Task

`process<T>` (declared in `sol.hpp`) is `noinline`, so the compiler is forced to emit a real, standalone symbol for every distinct `T` it's instantiated with. Fix `processAll` in `solve.cpp` so the three numeric values (`int`, `float`, `double`) are all converted to `double` *before* being passed to `process`, so they share exactly **one** instantiation (`process<double>`) instead of three. The two string values already share type `const char*`, so they're already fine as-is.

## Example

```cpp
// BAD: three distinct instantiations for three "numbers"
process<int>(i);
process<float>(f);
process<double>(d);

// GOOD: one shared instantiation
process<double>((double)i);
process<double>((double)f);
process<double>(d);
```

## What the gate checks

`main.cpp` calls `processAll()`, then shells out to `nm` on its **own** compiled executable and counts how many distinct `process<T>` symbols (mangled prefix `_Z7processI`) actually exist in the real binary -- not a simulated count. Correctly funneling all three numeric values through `process<double>` gives 2 distinct instantiations total (`double` + `const char*`); calling `process<int>`, `process<float>`, and `process<double>` separately gives 4. Your printed count is compared against `ref.cpp`, compiled and run the same way: `max_abs_err <= 1e-9`.
