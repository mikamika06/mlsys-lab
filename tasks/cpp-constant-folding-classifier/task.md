## Context

A C++ compiler's **constant-folding** pass evaluates expressions whose
value is knowable at compile time and replaces them with the literal
result, emitting no instructions. The precise, mechanical definition of
"foldable" is *whether the expression is a valid C++ core constant
expression* — the same rule the compiler uses to accept or reject an
expression where **only** a constant expression is legal, such as a
non-type template argument or an array bound.

You can turn that rule into a real compile-time test with a classic idiom:
wrap the expression in a lambda, and try to use a call to that lambda as
the initializer of a defaulted non-type template parameter:

```cpp
template <typename Lambda, int = (Lambda{}(), 0)>
constexpr bool is_constexpr_impl(Lambda) { return true; }
constexpr bool is_constexpr_impl(...) { return false; }
#define IS_FOLDABLE(...) is_constexpr_impl([]{ return (__VA_ARGS__); })
```

If `Lambda{}()` is a constant expression, the primary template is chosen
(`true`, folds). If not, substitution fails (SFINAE) and the `...`
fallback is chosen instead (`false`, does not fold) — the lambda body is
never actually executed in that case, so even runtime calls like `rand()`
or `std::time()` are safe to probe this way.

## Task

Implement, in `solve.cpp`,

```cpp
std::vector<int> classify_constant_folding();
```

Return, in order, one `1` (folds to a compile-time constant) or `0` (does
not) for each of these 12 fixed expressions:

```
 0. 2 + 3 * 4                          literal arithmetic
 1. sizeof(int)                        sizeof of a builtin type
 2. sizeof(int) * sizeof(double)       sizeof product
 3. 1 << 8                             literal bit-shift
 4. 0xFF & 0x0F                        literal bitwise AND
 5. __builtin_popcount(0xABCDu)        builtin applied to a literal
 6. { int arr[10]; sizeof(arr); }      sizeof of a fixed-size local array
 7. { constexpr int k = 7; k * 2; }    read of a constexpr local
 8. 1 / 0                              division by zero (UB)
 9. rand()                             runtime library call
10. std::sin(1.0)                      runtime floating-point library call
11. std::time(nullptr)                 runtime clock read
```

## Example

`2 + 3 * 4` is pure literal arithmetic — the compiler proves it at compile
time, so entry `0` is `1`. `rand()` depends on hidden runtime state (the
C library's PRNG state) that does not exist until the program runs, so
entry `9` is `0`.

## What the gate checks

The fixed driver (`main.cpp`) calls `classify_constant_folding()` and
prints whatever 0/1 vector comes back, space-separated, on one line. The
gate is an exact string match (`exact_match == 1.0`) against the
reference's printed line — the reference computes its answer with the
`IS_FOLDABLE` probe above against the real compiler, so it is not a
guessed answer key: every one of the 12 classifications must genuinely
match what this compiler proves foldable.
