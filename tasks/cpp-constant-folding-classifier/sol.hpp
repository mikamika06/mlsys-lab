#pragma once
#include <vector>

// ---------------------------------------------------------------------------
// LEARNER IMPLEMENTS.
//
// Classify each of the following 12 fixed C++ expressions as foldable to a
// compile-time constant (1) or not (0), IN THIS ORDER, and return the 12
// classifications as a vector<int>:
//
//    0. 2 + 3 * 4                              -- literal arithmetic
//    1. sizeof(int)                            -- sizeof of a builtin type
//    2. sizeof(int) * sizeof(double)           -- sizeof product
//    3. 1 << 8                                 -- literal bit-shift
//    4. 0xFF & 0x0F                            -- literal bitwise AND
//    5. __builtin_popcount(0xABCDu)            -- builtin applied to a literal
//    6. { int arr[10]; sizeof(arr); }          -- sizeof of a fixed-size local array
//    7. { constexpr int k = 7; k * 2; }        -- read of a constexpr local
//    8. 1 / 0                                  -- division by zero (UB, not foldable)
//    9. rand()                                 -- runtime library call
//   10. std::sin(1.0)                          -- runtime floating-point library call
//   11. std::time(nullptr)                     -- runtime clock read
//
// An expression "folds to a compile-time constant" exactly when it can be
// evaluated as a C++ core constant expression: the real, mechanical test is
// whether the compiler accepts it where ONLY a constant expression is
// legal (e.g. as a non-type template argument). If it does, entry i is 1;
// if using it there is ill-formed (a runtime call, UB such as division by
// zero, anything the compiler cannot resolve without running the program),
// entry i is 0.
// ---------------------------------------------------------------------------
std::vector<int> classify_constant_folding();
