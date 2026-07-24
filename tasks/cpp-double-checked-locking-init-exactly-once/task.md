## Context

**Double-checked locking** implements a thread-safe, lazily-initialized singleton without paying the cost of a mutex on every access. It works in two steps: first, a lock-free peek at whether the value is already initialized; only if not, take a lock and check *again* — because another caller may have finished initializing while this one was on its way to the lock — before actually initializing.

Skipping the second check is the classic bug: if two callers both observe "not ready" on the fast path before either reaches the lock, they will each take the lock in turn and, without a recheck, both perform the initialization.

## Task

`sol.hpp` declares a shared `SingletonState` (a real `std::atomic<bool> ready`, a real `std::mutex mtx`, and an `int init_count`). Implement:

```cpp
bool fast_check(const SingletonState& s); // lock-free peek at s.ready
bool try_init(SingletonState& s);         // take s.mtx, RE-CHECK, init if still needed
```

`try_init` must take `s.mtx`, check `s.ready` again now that the lock is held, and only increment `s.init_count` / set `s.ready` if it is still `false`. It must release the lock before returning (`std::lock_guard` is the natural tool here). Across any number of calls, `s.init_count` must end up exactly $1$.

## Example

If three callers all call `fast_check` before any of them calls `try_init` (all three see `ready == false`), then all three call `try_init` in turn: the first one finds `ready` still `false` under the lock, initializes, and sets `ready = true`. The second and third find `ready == true` under the lock and must skip initialization — even though their own `fast_check` earlier saw `false`.

## What the gate checks

`main.cpp` scripts exactly this worst-case race deterministically (no real OS threads, no timing): three `fast_check` calls, then three `try_init` calls, then prints every return value along with the final `init_count` and `ready`. The candidate's full stdout is compared byte-for-byte (`exact_match = 1.0`) against the reference's stdout. A `try_init` that forgets to recheck `s.ready` under the lock initializes three times instead of once — the exact real-world double-checked-locking bug — and a `try_init` that forgets to release the lock deadlocks on its second call and times out.
