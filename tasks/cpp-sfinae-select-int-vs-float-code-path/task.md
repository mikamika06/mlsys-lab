## Context

**SFINAE** ("Substitution Failure Is Not An Error") lets overload
resolution pick between template overloads based on a type trait, entirely
at **compile time**. `std::enable_if<Condition, R>::type` only exists as a
type when `Condition` is `true`; when it's `false`, substituting `T` into
that overload's template parameters fails — and instead of a hard compile
error, the compiler just quietly removes that overload from the candidate
set and looks at the others.

```cpp
template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
int classify_impl(T);   // a candidate only when T is integral

template <typename T, typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
int classify_impl(T);   // a candidate only when T is floating-point
```

For any concrete `T`, exactly one of these two is ever a viable candidate —
the *other* overload is never even instantiated for that `T`. This is a
fundamentally different mechanism from writing one generic function body and
branching on `std::is_integral<T>::value` at runtime: here, the "branch"
happens during overload resolution, before any code runs.

## Task

Implement

```cpp
template <typename T>
int classify(T x);
```

so it returns `0` when `T` is an integral type, `1` when `T` is a
floating-point type — using two SFINAE/`enable_if`-constrained overloads
(as shown above) to make the compiler select the right one, not a runtime
`if (std::is_integral<T>::value)` inside a single template body.

`classify` is defined in `solve.cpp` (not `sol.hpp`), so it needs an
**explicit template instantiation** for every `T` the driver calls it with
— `template int classify<int>(int);` and so on — right after the
definition, so the linker can find the code `main.cpp` calls.

## Example

```cpp
classify<int>(5);      // -> 0  (int is integral)
classify<double>(1.5); // -> 1  (double is floating-point)
```

## What the gate checks

The driver calls `classify<T>` for six types (`int`, `long`, `char`,
`unsigned`, `float`, `double`) and prints the tag returned for each. The
grader compiles `solve.cpp` with `clang++ -O2 -std=c++20`, runs it, and
requires

$$
\mathrm{exact\_match} = 1 \iff \text{every printed tag matches the reference}
$$

Returning a fixed tag regardless of `T` gets the integral types right by
coincidence but is wrong for `float`/`double` — the gate needs the compiler
to have genuinely selected a *different* overload for those two calls, not
just a function that happens to return the right constant sometimes.
