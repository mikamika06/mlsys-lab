import sys

def _fib(n: int) -> int:
    """Naive recursive Fibonacci used by the oracle."""
    if n <= 1:
        return n
    return _fib(n - 1) + _fib(n - 2)

def _ref_count(n: int) -> int:
    """Expected number of calls during fib(n) = 2*F_{n+1} - 1."""
    return 2 * _fib(n + 1) - 1

def grade(sol, fx) -> dict:
    test_cases = [0, 1, 2, 3, 5, 8]
    ok = 1.0
    for n in test_cases:
        try:
            cnt = sol.count_fib_frames(n)
        except Exception:
            ok = 0.0
            break
        if cnt != _ref_count(n):
            ok = 0.0
            break
    return {"exact_match": ok}
