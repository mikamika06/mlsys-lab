## Context

Tracking peak live-object count and peak memory footprint matters for
catching unbounded heap growth and allocation spikes under a workload. The
fixed instrumented probe (`sol.hpp`) does the bookkeeping FOR you, but only
if you actually construct and destroy real objects:

```cpp
struct Probe {
    int x; double y;
    Probe();   // ++g_live; g_peak_live = max(g_peak_live, g_live)
    ~Probe();  // --g_live
};
```

`g_peak_live` is the high-water mark of `g_live` ever observed — but it can
only see real `Probe` constructions and destructions. Tracking counts by
hand without ever building a real `Probe` leaves `g_peak_live` at `0` no
matter what the workload says.

## Task

Implement `run_workload(const int* ids, const bool* is_alloc, int n)`
(declared in `sol.hpp`) in `solve.cpp`. For each of the `n` events, in
order:

- `is_alloc[i]` true — allocate: `new Probe()`, keyed by `ids[i]` (so you
  can find it again later).
- `is_alloc[i]` false — free: `delete` whatever object is currently live
  under `ids[i]`, if any.

You'll need your own id -> `Probe*` map (e.g. `std::map<int, Probe*>`),
local to your implementation.

## Example

For the fixed driver's four workloads, the correct run prints (peak live
count, then `peak_live * sizeof(Probe)`):

```
3 48
1 16
5 80
2 32
```

The first workload — `alloc 1, alloc 2, alloc 3, free 1, alloc 4, free 2` —
peaks at 3 live objects (right after `alloc 3`; the later `alloc 4` only
brings it back up to 3, not past it). A starter that never constructs a real
`Probe` leaves `g_peak_live` at `0` for every workload: `0 0` four times.

## What the gate checks

The grader compiles `main.cpp` + `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires an **exact match** of all four printed
`peak_live peak_memory_bytes` pairs against the same driver linked with
`ref.cpp`. Freeing an id that was never allocated, double-freeing, or
tracking counts without touching real `Probe` objects all produce a wrong
(or stuck-at-zero) peak and fail the gate.
