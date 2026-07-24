## Context

A **data race** happens when two threads access the same memory location
concurrently, at least one access is a write, and nothing establishes an
ordering between them. Wrapping a variable in `std::atomic<T>` does not, by
itself, make *every* sequence of operations on it race-free — it only
guarantees that each *individual* `load()` or `store()` is indivisible.

A very common mistake:

```cpp
long v = counter.value.load();   // step 1: read
v = v + 1;                       // step 2: compute, in a local (not shared)
counter.value.store(v);          // step 3: write back
```

Each of steps 1 and 3 is atomic on its own, but the *sequence* is not: two
threads can both execute step 1 and read the same old value before either
one reaches step 3. Both then compute `old + 1` and both store the same
result back — one of the two increments is silently lost. This is exactly a
classic check-then-act (TOCTOU) race, just dressed up with an atomic type.

The fix is to perform the whole increment as a single **read-modify-write**
instruction, which the hardware executes indivisibly — no other thread's
RMW can interleave in the middle of it:

```cpp
counter.value.fetch_add(1, std::memory_order_relaxed);
```

## Task

Fix `increment(SharedCounter& counter)` (declared in `sol.hpp`) so it
increments `counter.value` by exactly 1, safely, under real concurrent
callers — using a single atomic read-modify-write instead of a separate
load and store.

## Example

The driver spawns 8 threads that each call `increment()` 200000 times on
the *same* `SharedCounter`, with no synchronization between calls other than
`increment` itself, then joins all of them and reads the final value. A
correct RMW increment can never lose an update — the atomicity guarantee
does not depend on scheduling — so the final count is **exactly**
`8 * 200000 = 1600000` on every single run, regardless of how the OS
interleaves the threads.

## What the gate checks

The driver prints `final=<n> expected=<n> ok=<0|1>`. The grader compiles
`solve.cpp` with `clang++ -O2 -std=c++20`, runs it (spawning 8 real
`std::thread`s), and requires

$$
\mathrm{exact\_match} = 1 \iff \text{the printed final count equals } T \times \mathrm{ITERS}
$$

A load-then-store "increment" loses updates under real concurrent execution
at this scale — not occasionally, but on essentially every run, since 1.6
million increments across 8 real threads give the race an enormous number of
chances to interleave badly. The gate is on the actual printed count from
actually running the threads, not a simulated or predicted one.
