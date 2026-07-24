## Context

C++ `constexpr` functions can be evaluated by the compiler when their inputs are known at compile time. That only works when the function body is deterministic, dependency-free integer math — no floating point, no I/O, no dynamic allocation. This task is exactly that kind of algorithm: an integer square root and a primality test, both restricted to integer-only arithmetic.

For an integer $n \ge 0$, the integer square root is the largest integer $r$ such that

$$r^2 \le n.$$

A primality test marks an integer $n > 1$ as prime when there is no divisor $d$ satisfying

$$2 \le d \le \lfloor \sqrt{n} \rfloor \quad \text{and} \quad n \bmod d = 0.$$

Using the square-root boundary is sufficient because any composite number must have at least one factor not greater than its square root.

## Task

Implement:

```cpp
int integer_sqrt(int n);
bool is_prime(int n);
```

`integer_sqrt(n)` returns the largest integer $r$ with $r^2 \le n$, for $n \ge 0$. `is_prime(n)` returns whether $n$ is prime, using `integer_sqrt` to bound the trial-division loop. Neither function may use `<cmath>` or any floating-point operation — the whole point is that this algorithm stays within the integer domain a real `constexpr` function is restricted to.

## Example

`integer_sqrt(10)` is `3` (since $3^2=9\le10$ but $4^2=16>10$). `is_prime(10)` is `false` (divisible by `2`). `integer_sqrt(17)` is `4`, and `is_prime(17)` is `true` (no divisor in `[2, 4]`).

## What the gate checks

`main.cpp` calls both functions over a fixed set of $24$ inputs spanning small values, primes, perfect squares, and larger composites, printing `integer_sqrt(v)`, `is_prime(v)`, and `v` for each. The candidate's full stdout is compared byte-for-byte (`exact_match = 1.0`) against the reference's stdout. A floating-point `sqrt`-based shortcut that rounds wrong on even one boundary value (perfect squares are the classic failure point) produces a different trace and fails the gate.
