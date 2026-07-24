## Context

The **detection idiom** (SFINAE + `std::void_t` + `decltype`) introspects a
type at compile time: does a given expression involving it compile at all?
A common use is asking whether `T` has a member function — here,
`serialize` — callable with a specific argument type.

```cpp
template <typename T, typename = void>
struct has_serialize : std::false_type {};

template <typename T>
struct has_serialize<T, std::void_t<
    decltype(std::declval<T>().serialize(std::declval<int>()))
>> : std::true_type {};
```

The primary template is the fallback (`value == false`). The partial
specialization is only a viable match when its `std::void_t<...>` argument
is well-formed — i.e. when `std::declval<T>().serialize(std::declval<int>())`
actually compiles for `T`. `std::declval<int>()` (not a literal `0`) is used
deliberately: it is not a constant expression, so it can never be treated as
a null-pointer constant, which keeps "does `int` convert to this parameter?"
honest for pointer parameters (it never does).

For that expression to compile, overload resolution must find a `serialize`
overload callable with exactly one argument (its required-argument count
$\le 1 \le$ its total argument count, counting defaults), whose first
parameter accepts an `int` by implicit conversion — true for the other
numeric primitives, never true for a pointer parameter.

`sol.hpp` fixes twelve probe types (`DProbe1`..`DProbe12`), each real C++,
each exercising one edge case: no `serialize` at all, wrong arity, a
pointer-only overload, a convertible overload, an overload set mixing 0-arg
and 1-arg versions, default arguments that do or don't rescue arity, etc.

## Task

Implement, in `solve.cpp`, the twelve functions declared in `sol.hpp`
(`detect_DProbe1()` .. `detect_DProbe12()`), each returning whether its
`DProbeN` satisfies the detection idiom above.

Define the `has_serialize<T>` trait — primary template plus the
`std::void_t`-gated partial specialization — **once, in `solve.cpp`**, then
have each `detect_DProbeN()` return `has_serialize<DProbeN>::value`. (A
template's specializations must be visible, in the same translation unit,
everywhere they get instantiated — that's why this trait can't live split
across the shared header and `solve.cpp`; it has to be defined and
instantiated together, in `solve.cpp`.)

## Example

For the fixed driver's twelve probes, the correct run prints:

```
0 0 1 0 1 0 1 1 0 1 1 0
```

in `DProbe1`..`DProbe12` order:

- `DProbe1` — no `serialize` method at all -> `0`.
- `DProbe2` — `serialize()` takes 0 args, can't be called with 1 -> `0`.
- `DProbe3` — `serialize(int)` -> `1`.
- `DProbe4` — `serialize(int, int)` requires 2 -> `0`.
- `DProbe5` — `serialize(int, double = 0.0)` requires 1, accepts up to 2 -> `1`.
- `DProbe6` — `serialize(void*)`: `int` doesn't convert to a pointer -> `0`.
- `DProbe7` — `serialize(double)`: `int` converts to `double` -> `1`.
- `DProbe8` — overload set `serialize()` / `serialize(int)`: the 1-arg one matches -> `1`.
- `DProbe9` — `serialize(char*)`: pointer -> `0`.
- `DProbe10` — `serialize(int=0, int=0, int=0)`: 0 required, 1 is in range -> `1`.
- `DProbe11` — `serialize(int=0)`: 0 required, 1 accepted -> `1`.
- `DProbe12` — `serialize(void*, int=0)`: arg0 is still a pointer -> `0`.

The starter has no `has_serialize` trait at all and every `detect_DProbeN()`
hardcodes `false`, so it prints `0 0 0 0 0 0 0 0 0 0 0 0` — wrong on the six
probes that should detect `true`.

## What the gate checks

The grader compiles `main.cpp` + `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires `max_abs_err <= 1e-6` against the same driver linked
with `ref.cpp`. A trait that's too permissive (e.g. accepting pointer
arguments) or too strict (e.g. missing the overload-set or default-argument
cases) misclassifies at least one probe and fails the gate — real overload
resolution decides the answer, not a modeled rule.
