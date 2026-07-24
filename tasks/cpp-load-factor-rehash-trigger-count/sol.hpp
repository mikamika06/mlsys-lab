#pragma once

struct HashNode {
    void* next;
    int key_hash;
    long key;
    double value;
};

struct SimResult {
    int rehash_count;
    int hash_node_size; // your prediction of sizeof(HashNode) under LP64
};

// Simulate inserting `n` long keys (in order, `inserts[0..n)`) into a hash
// map that starts with `initial_buckets` buckets. Inserting a key already
// present only updates its value: size does not change and no rehash can
// happen. Inserting a NEW key increases size by 1; if that would make
// size / bucket_count strictly exceed max_load_factor, a rehash happens
// FIRST (bucket_count doubles, rehash_count += 1), then the key is added.
SimResult simulate_hash_map(const long* inserts, int n, double max_load_factor, int initial_buckets);
