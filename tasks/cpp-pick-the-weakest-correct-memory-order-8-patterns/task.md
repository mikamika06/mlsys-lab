## Context

Choosing the weakest sufficient `std::memory_order` avoids unnecessary memory fences while still guaranteeing correctness under the real C++ memory model. In order of strength:

$$\text{relaxed} \;<\; \text{acquire} / \text{release} \;<\; \text{acq\_rel} \;<\; \text{seq\_cst}$$

Each synchronization role has a specific minimum requirement:

1. **counter_increment**: a standalone counter increment, nothing else depends on the ordering $\to$ `relaxed`.
2. **publish_store**: writing a payload, then setting a "ready" flag $\to$ `release` (so the writes happen-before the flag is observed).
3. **consume_load**: reading a "ready" flag, then reading the payload $\to$ `acquire` (so the reads happen-after the flag publishes).
4. **lock_acquire**: acquiring a spinlock via CAS/`test_and_set` $\to$ `acquire`.
5. **lock_release**: releasing a spinlock via store/clear $\to$ `release`.
6. **rmw_sync**: a read-modify-write that must both acquire and release (e.g. a refcount decrement guarding a destructor) $\to$ `acq_rel`.
7. **total_order**: needs one global total order across every thread (e.g. Dekker's algorithm / a store-buffering litmus test) $\to$ `seq_cst`.
8. **relaxed_read**: a read with no synchronization requirement at all $\to$ `relaxed`.

## Task

`OpRole` and `std::memory_order` (the real enum from `<atomic>`) are declared in `sol.hpp`. Fix `weakestOrderFor` in `solve.cpp` so it returns the correct order for every role, per the list above. The shipped version has `acquire` and `release` swapped on the four direction-sensitive roles (`publish_store`, `consume_load`, `lock_acquire`, `lock_release`) -- a very common real mixup, since "acquire" and "release" sound almost interchangeable but mean opposite happens-before directions.

## Example

```cpp
weakestOrderFor(OpRole::PublishStore);  // must return std::memory_order_release
weakestOrderFor(OpRole::ConsumeLoad);   // must return std::memory_order_acquire
```

## What the gate checks

`main.cpp` calls `weakestOrderFor` for all 8 roles and prints each role's name alongside `(int)` the returned `std::memory_order` value. Your printed output is compared against `ref.cpp`, compiled and run the same way: `max_abs_err <= 1e-9` (the standard guarantees `memory_order`'s enumerators are all distinct values, so any swap shows up as a numeric mismatch). Swapping acquire and release on any of the four direction-sensitive roles breaks that role's line while leaving the other four untouched.
