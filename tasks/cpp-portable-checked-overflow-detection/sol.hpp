#pragma once

// ============================================================================
// LEARNER implements these twelve in solve.cpp.
//
// Each returns whether `a OP b`, computed as if in the named signed type,
// would overflow that type's range — WITHOUT ever triggering the overflow
// as undefined behaviour. Signed integer overflow (`INT_MAX + 1`, etc.) is
// UB in C++: you may never compute `a + b` (or `-`, `*`) directly in the
// narrow signed type when it might overflow. Compute the operation in a
// wider type that is guaranteed large enough to hold any possible result
// (e.g. `__int128` for every case here, since it comfortably holds the
// product of any two 64-bit values), then range-check the wide result
// against the narrow type's real `min`/`max`.
// ============================================================================
bool add_overflow_int(int a, int b);
bool sub_overflow_int(int a, int b);
bool mul_overflow_int(int a, int b);

bool add_overflow_short(short a, short b);
bool sub_overflow_short(short a, short b);
bool mul_overflow_short(short a, short b);

bool add_overflow_char(signed char a, signed char b);
bool sub_overflow_char(signed char a, signed char b);
bool mul_overflow_char(signed char a, signed char b);

bool add_overflow_long(long a, long b);
bool sub_overflow_long(long a, long b);
bool mul_overflow_long(long a, long b);
