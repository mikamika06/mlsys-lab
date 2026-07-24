## Context

**False sharing** happens when independent variables owned by different
threads land on the same cache line (typically 64 bytes). Even though the
threads share no data, the cache-coherence protocol still bounces that line
between cores on every write, killing performance.

```cpp
struct ThreadData { long counter; };
ThreadData data[4];
```

`sizeof(ThreadData) == 8` — all 4 counters fit in the first 32 bytes,
meaning all 4 land in the very first 64-byte cache line. Thread 1 writing
`data[1].counter` and thread 2 writing `data[2].counter` are false-sharing
writes even though they touch completely different variables. The usual fix
is padding the struct so its size is a multiple of the cache line size.

Real threads and wall-clock timing aren't reproducible, so the fixed driver
(`main.cpp`) instead uses a deterministic proxy: treating `data[4]` as
starting at address 0, it walks threads `0..3` in order and, for each one,
checks whether the 64-byte line containing `data[thread_id].counter` is
already "owned" by a **different** thread — if so, that's a false-sharing
write.

## Task

Define, in `solve.cpp`:

```cpp
struct ThreadData {
    long counter;   // must stay the FIRST member
    /* your padding fields, appended after counter */
};
```

and implement `thread_data_sizeof()` (declared in `sol.hpp`) to return
`sizeof(ThreadData)` — the real compiler's answer. Add enough padding after
`counter` that `sizeof(ThreadData)` becomes a multiple of 64, so
`thread_id * sizeof(ThreadData)` never lands two different threads' counters
in the same cache line.

## Example

For the fixed driver, the correct run (`sizeof(ThreadData) == 64`) prints:

```
stride=64
shared_writes=0
```

The unpadded starter (`sizeof(ThreadData) == 8`) puts all 4 counters at
addresses `0, 8, 16, 24` — the same line as thread 0's — for `3`
false-sharing writes:

```
stride=8
shared_writes=3
```

## What the gate checks

The grader compiles `main.cpp` + `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires an **exact match** of the printed stride and
shared-write count against the same driver linked with `ref.cpp`. Padding to
anything short of a multiple of 64 (e.g. 32 or 48 bytes) still leaves at
least one pair of threads sharing a line and fails the gate.
