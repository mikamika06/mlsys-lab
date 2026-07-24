## Context

`std::atomic<long>` with `fetch_add` guarantees that every increment is applied
as an indivisible read-modify-write. No matter how $T$ real OS threads interleave
their $I$ increments each — however the scheduler slices CPU time between them —
the final value is always exactly

$$\text{final} = T \times I$$

A plain, non-atomic counter shared by multiple threads would instead race: two
threads can both read the same old value, both add 1, and both write back the
same new value, silently losing an increment. `std::atomic` closes that window
by making the read-modify-write a single hardware-guaranteed atomic operation.

## Task

Implement

```cpp
long atomic_counter_final_value(int num_threads, int increments_per_thread);
```

- Spawn `num_threads` real `std::thread` workers.
- Each worker calls `fetch_add(1, ...)` on **one shared** `std::atomic<long>`
  counter, exactly `increments_per_thread` times.
- Join every thread.
- Return the counter's final value.

Because the increments are atomic, the result is deterministic and does not
depend on thread scheduling, core count, or timing.

## Example

```cpp
atomic_counter_final_value(4, 1000);   // -> 4000
atomic_counter_final_value(1, 500);    // -> 500
atomic_counter_final_value(8, 125);    // -> 1000
```

## What the gate checks

The driver runs your function on several `(num_threads, increments_per_thread)`
pairs and prints each returned value. The grader compiles `solve.cpp` with
`clang++ -O2 -std=c++20`, runs it, and requires

$$ \mathrm{exact\_match} = 1.0 $$

against the reference, which computes each final value by actually running
`num_threads` real threads to completion on a real `std::atomic<long>`. Every
printed value must equal `num_threads * increments_per_thread` exactly — an
implementation that forgets to spawn threads, forgets to join them before
reading the counter, or uses a non-atomic counter under real concurrent access,
will not reliably reproduce that product.
