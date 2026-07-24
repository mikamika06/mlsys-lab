## Context

In CPython, every function call creates a *frame object* that holds the execution state (local variables, bytecode pointer, etc.). The function `sys.setprofile(prof_func)` installs a callback that is invoked for each call/return event in the current thread. The callback receives `(frame, event, arg)` where `event` is `'call'` for a function invocation, `'return'` for a return, and `'c_call'` / `'c_return'` for C calls. By counting only the `'call'` events, we can measure how many stack frames are created during a given computation.

The classic naive recursive Fibonacci function

$$F_n = \begin{cases}0 & n = 0\\ 1 & n = 1\\ F_{n-1}+F_{n-2} & n \ge 2\end{cases}$$

produces a recursion tree whose total number of calls (including the root) equals $2F_{n+1} - 1$.  For example, $n=2$ gives $2F_3 - 1 = 2\cdot 2 - 1 = 3$ calls.

## Task

Implement the function `count_fib_frames(n)` that returns the exact number of Python stack frames created — i.e., the total number of `'call'` events — during the evaluation of `fib(n)` using the naive recursive definition above. You must use `sys.setprofile` to observe the events.

**Specification:**
- `n` is a non‑negative integer.
- The return value must be an integer equal to the total call count.

**Constraints:**
- You are not allowed to alter the definition of `fib` beyond the naive recursion.
- Do **not** count calls to `sys.setprofile` or the profile function itself — only the calls to `fib`.

## Example

```python
>>> count_fib_frames(2)
3
>>> count_fib_frames(3)
5
```

Explanation: `fib(2)` calls `fib(1)` and `fib(0)`, making 3 frames. `fib(3)` calls `fib(2)` and `fib(1)`, and the subtree of `fib(2)` produces 3 calls, totaling 5.

## What the gate checks

The gate compares your returned count to the analytically computed expected count $2F_{n+1} - 1$ for several small values of $n$ (0, 1, 2, 3, 5, 8). All counts must match **exactly** for the gate to pass.
