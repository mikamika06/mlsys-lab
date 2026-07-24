## Context

`std::unordered_map` is typically implemented as an array of buckets, each pointing to a linked list of nodes. `sol.hpp` gives a representative node struct:

```cpp
struct HashNode {
    void* next;
    int key_hash;
    long key;
    double value;
};
```

As elements are inserted, the map tracks its **load factor** ($\text{size} / \text{bucket\_count}$). If an insertion would make the load factor *strictly exceed* `max_load_factor`, the map **rehashes** first — in this simplified model, that exactly doubles `bucket_count`.

## Task

Implement `SimResult simulate_hash_map(const long* inserts, int n, double max_load_factor, int initial_buckets)`:

1. Walk `inserts[0..n)` in order. If a key was already seen, only its value updates — size and bucket count are untouched, no rehash.
2. If a key is new: check whether `size + 1` would make the load factor strictly exceed `max_load_factor` against the *current* `bucket_count`. If so, rehash first (`rehash_count += 1`, `bucket_count *= 2`) — possibly needing only one doubling, since the check uses the *current* bucket count each time. Then add the key (`size += 1`).
3. Set `hash_node_size` to your own prediction of `sizeof(HashNode)` under LP64 rules (natural alignment, padding, no reordering of fields).

## Example

`inserts = [10, 20, 30, 40, 50, 10]`, `max_load_factor = 1.0`, `initial_buckets = 4`: the first four inserts fill the map to `size = 4`, `buckets = 4` (load `1.0`, not yet over). Before inserting `50`, size would become `5`; $5/4 = 1.25 > 1.0$, so a rehash happens (`buckets` → `8`, `rehash_count` → `1`), then `50` is added. Re-inserting `10` just updates it — no rehash. Final `rehash_count = 1`.

## What the gate checks

`main.cpp` runs four fixed insertion sequences (including repeated keys, a large monotonically-growing sequence, and an all-duplicate sequence) through `simulate_hash_map` and prints `rehash_count`, `hash_node_size`, and whether `hash_node_size` matches the compiler's real `sizeof(HashNode)`. The candidate's full stdout is compared byte-for-byte (`exact_match = 1.0`) against the reference's. Rehashing on the current size instead of the size *after* the pending insert, or checking `>=` instead of strictly `>`, both shift exactly when a rehash fires and change `rehash_count` on at least one of the four sequences.
