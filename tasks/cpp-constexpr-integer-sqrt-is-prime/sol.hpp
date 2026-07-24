#pragma once
// Both functions must be written using integer-only arithmetic — no
// <cmath>, no floating-point square root, no floating-point operations of
// any kind. This is exactly the kind of algorithm real `constexpr`
// functions rely on: deterministic, dependency-free integer math that a
// compiler is able to evaluate entirely at compile time.

// Largest integer r with r*r <= n, for n >= 0.
int integer_sqrt(int n);

// True iff n > 1 and n has no divisor d with 2 <= d <= floor(sqrt(n)).
bool is_prime(int n);
