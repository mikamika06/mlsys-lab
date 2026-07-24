## Context

`std::decay<T>::type` (from `<type_traits>`) models the type transformations that happen when you pass an argument by value to a function parameter:

1. If `T` is a reference (`T&` or `T&&`), decay the referenced type.
2. Else if `T` is an array type `Elem[N]` (or `Elem[]`), it decays to `Elem*`.
3. Else if `T` is a function type `Ret(Args...)`, it decays to a function pointer `Ret(*)(Args...)`.
4. Otherwise, strip only the **top-level** `const`/`volatile`. A pointee's own qualifiers are not top-level and survive: `const int* const` decays to `const int*`, not `int*`.

## Task

Implement `template<typename T> struct MyDecay` from scratch in `solve.cpp` (a primary template plus the partial specializations needed for references, arrays, and function types), matching `std::decay_t<T>` for every one of the 15 concrete types listed in `sol.hpp`. Each `checkTypeN()` must return `true` iff `std::is_same<typename MyDecay<T>::type, std::decay_t<T>>::value` holds for that function's `T`.

The shipped `MyDecay` is just the identity (`using type = T;`) -- no specializations at all, so it only happens to be correct for plain `int`.

## Example

```cpp
template <typename T> struct MyDecay<T&> { using type = typename MyDecay<T>::type; };
template <typename T, std::size_t N> struct MyDecay<T[N]> { using type = T*; };
template <typename Ret, typename... Args> struct MyDecay<Ret(Args...)> { using type = Ret(*)(Args...); };
// primary template (falls through here for everything else):
template <typename T> struct MyDecay { using type = typename std::remove_cv<T>::type; };
```

## What the gate checks

`main.cpp` calls `checkType1()` through `checkType15()` and prints each result. `ref.cpp`'s `MyDecay` is checked against the real `std::decay_t` directly, so it is correct by construction -- every reference line reads `1`. Your printed output is compared against it, compiled and run the same way: `max_abs_err <= 1e-9`. The identity starter passes only `type1` (plain `int`, which happens to already equal its own decay); every reference, array, function, and cv-qualified type fails.
