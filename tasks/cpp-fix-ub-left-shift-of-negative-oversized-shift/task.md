## Context

In C++, left-shifting a negative signed integer, or shifting by an amount
greater than or equal to the bit-width of the type, is **Undefined Behavior
(UB)**. Consider the naive operation:

```cpp
int value = ...;
int shift_amount = ...;
long result = (long)(value << shift_amount);  // UB if value < 0 or shift_amount >= 32!
```

This is not a theoretical concern: at `-O2`, the compiler is free to assume
UB never happens, and can legally produce a result that does not match
"the two's-complement bit pattern shifted the obvious way" -- exactly the
answer a naive port from another language, or careless mental model, would
expect.

To perform the shift safely and get the well-defined, portable,
two's-complement answer, you must:
1. Reinterpret `value` as `unsigned int` (a bitcast — the unsigned type has
   no shift-of-negative UB).
2. Clamp `shift_amount` into `[0, 31]` by taking it modulo 32 (unsigned
   shift by an in-range amount is always well-defined; the top bits are
   simply discarded).
3. Perform the shift on the unsigned value.
4. Reinterpret the 32-bit unsigned result back as a signed `int`, then store
   it into the 64-bit `long` result, which sign-extends it.

## Task

Implement

```cpp
void process_shifts(const int* values, const int* shift_amounts, long* results, int n);
```

For each `i` in `[0, n)`, compute the safe equivalent described above of
`value[i] << shift_amounts[i]` and store it in `results[i]`.

## Example

```
value = -1, shift_amount = 31
(unsigned int)(-1)        = 0xFFFFFFFF
0xFFFFFFFF << 31          = 0x80000000
(int)0x80000000           = -2147483648
stored as long (sign-ext) = -2147483648
```

## What the gate checks

The driver runs `process_shifts` over 8 fixed cases mixing negative values,
positive values, in-range shifts, and shifts `>= 32`, and prints each
`results[i]`. The grader compiles `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires

$$ \mathrm{exact\_match} = 1.0 $$

against the reference. A kernel that performs the raw `value << shift_amount`
on signed `int` triggers real undefined behavior at `-O2` on several of the
oversized-shift cases — the optimizer legitimately produces `0` instead of
the two's-complement answer for those entries, so the printed trace stops
matching the moment UB is present.
