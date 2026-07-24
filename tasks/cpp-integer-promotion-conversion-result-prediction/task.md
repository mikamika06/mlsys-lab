## Context

Binary arithmetic ($+$, $-$, $*$) between two different integer types in C++ follows strict rules: **integer promotion**, then the **usual arithmetic conversions**. On this ABI: `char`/`unsigned char` = 1 byte (`char` is signed here), `short`/`unsigned short` = 2 bytes, `int`/`unsigned int` = 4 bytes, `long`/`unsigned long` = 8 bytes.

1. **Promotion**: any operand narrower than `int` (`char`, `unsigned char`, `short`, `unsigned short`) is promoted to `int` first -- `int`'s full 32-bit range covers all of their values.
2. **Usual arithmetic conversions**, applied to the (now `int`-or-wider) operand types if they still differ:
   - same signedness -> the higher-rank type wins (rank order `int`/`unsigned int` < `long`/`unsigned long`)
   - different signedness, same rank, or the unsigned operand's rank is $\geq$ the signed operand's rank -> the **unsigned** type wins
   - different signedness, the signed operand's rank is strictly higher AND it can represent the unsigned operand's whole range (true for `long` vs `unsigned int` on this ABI) -> the **signed** type wins
3. The operation is evaluated in that common type. Unsigned results wrap modulo $2^{\text{width}}$ (always well-defined); signed overflow is undefined behavior in C++, so every expression here is chosen so the signed path never overflows -- only the (perfectly well-defined) unsigned wraparound is exercised.

## Task

Implement `evalConversion(op, lhsType, lhsVal, rhsType, rhsVal)` in `solve.cpp`, per the contract and rule summary in `sol.hpp`: work out the common result type from `lhsType`/`rhsType` following the rules above, evaluate `op` in that type, and return the result's value (as a bit pattern -- signed directly, unsigned via `(long long)(unsigned long long)value`), its `width` (`sizeof`, 4 or 8), and whether it `isSigned`.

## Example

```cpp
evalConversion('+', IntType::Char, 100, IntType::Short, 200);
// char + short: both promote to int -> (300, width=4, signed=true)

evalConversion('+', IntType::Int, -1, IntType::UInt, 1);
// int + unsigned int: unsigned wins -> (-1 as unsigned) + 1 wraps to 0
// -> (0, width=4, signed=false)
```

## What the gate checks

`ref.cpp` never hand-simulates the rule table: it evaluates each expression with the **actual native C++ types** (`(char)100 + (short)200`, etc.) and reads the result's real type back via `decltype` -- the real compiler's promotion and conversion rules are the oracle. `main.cpp` runs 16 expressions covering same-signed promotion, mixed-signed same-rank, mixed-signed different-rank (both directions), and unsigned wraparound on subtraction, addition, and multiplication, and prints each result's value/width/signedness. Your printed output is compared against `ref.cpp`, compiled and run the same way: `max_abs_err <= 1e-9`. Picking the wrong common type on any mixed-signedness case, or forgetting that small types promote all the way to `int` (not to their own unsigned variant), throws off the corresponding line.
