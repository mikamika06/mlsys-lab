## Context

A **bounded queue** coordinated via `std::mutex` (and, in a real multi-threaded program, `std::condition_variable`) lets producers and consumers safely exchange data while enforcing a capacity bound. Two invariants must always hold:

1. **Capacity boundary**: at every instant, the queue length $L$ satisfies $0 \le L \le C$.
2. **Conservation of payload sum**: whatever was pushed and never popped is still sitting in the queue — nothing is created or lost:
$$\sum \text{consumed payloads} = \sum \text{produced payloads} - \sum_{k \in \text{queue}} \text{payload}_k.$$

## Task

`sol.hpp` declares a real, mutex-protected `BoundedQueue` of `Item{int key; double payload;}`. Implement:

```cpp
void bq_init(BoundedQueue& q, int capacity);
void bq_push(BoundedQueue& q, Item item, double* produced_sum);
void bq_pop(BoundedQueue& q, double* consumed_sum);
```

`bq_push`/`bq_pop` must each lock `q.mtx` for their entire critical section. `bq_push` drops the item (does nothing beyond returning) if the queue is already at `capacity`; otherwise it appends and adds `item.payload` to `*produced_sum`. `bq_pop` drops the request if the queue is empty; otherwise it removes the front item (FIFO) and adds its payload to `*consumed_sum`.

## Example

With `capacity = 2`: push `(10, 1.5)`, push `(20, 2.5)`, push `(30, 3.5)` (dropped — already at capacity `2`), pop (removes `(10, 1.5)`), push `(30, 3.5)` (now fits), pop, pop. Every pushed item that was ever accepted eventually gets popped, so `produced_sum == consumed_sum == 7.5` and the queue ends empty.

## What the gate checks

`main.cpp` scripts four fixed push/pop sequences (single-threaded, fully deterministic — no real OS threads, no timing) against real `BoundedQueue`s of varying capacity, including one that ends with an item still in the queue, and prints `produced_sum`, `consumed_sum`, and the final `count` for each. The candidate's full stdout is compared byte-for-byte (`exact_match = 1.0`) against the reference's. Failing to check capacity before pushing lets the queue silently overflow its fixed backing array; failing to check emptiness before popping corrupts `count`.
