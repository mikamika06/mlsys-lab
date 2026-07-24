## Context

**False sharing** occurs when two or more threads access different
variables that happen to reside on the same cache line. If at least one
thread is writing to that line, the CPU's cache coherence protocol keeps
invalidating and transferring the line between cores, severely degrading
performance — even though the threads never touch the same variable.

Assume a standard 64-byte cache line, and that a `ThreadState` instance is
allocated at an address aligned to 64 bytes (so byte offset `o` falls in
cache line `o / 64`):

```cpp
struct ThreadState {
    int       id;
    int       read_count;
    long      write_count;
    double    items[5];
    long long padding[5];
    double    local_sum;
    int       local_flag;
};
```

## Task

Implement, in `solve.cpp`,

```cpp
std::pair<std::vector<bool>, long> classify_false_sharing();
```

Classify these 10 fixed concurrent access patterns (Thread 1 and Thread 2
both touching the same `ThreadState` instance simultaneously):

```
 1. T1 writes id,          T2 writes read_count
 2. T1 writes id,          T2 reads  write_count
 3. T1 reads  read_count,  T2 reads  write_count
 4. T1 writes items[4],    T2 writes padding[0]
 5. T1 writes items[4],    T2 writes padding[1]
 6. T1 writes padding[0],  T2 writes padding[1]
 7. T1 writes local_sum,   T2 writes local_flag
 8. T1 writes items[0],    T2 reads  local_sum
 9. T1 writes padding[1],  T2 reads  local_flag
10. T1 reads  items[0],    T2 writes items[1]
```

A pattern is false sharing exactly when both accesses fall in the **same**
64-byte cache line **and** at least one of the two is a write. Two
simultaneous reads never cause false sharing; accesses landing in
different lines never do either, however many writes are involved.

Return `{labels, struct_size}`: `labels` is exactly 10 booleans in the
order above (`true` = false sharing), and `struct_size` is
`sizeof(ThreadState)`. Find each field's real byte offset with
`offsetof(ThreadState, field)` on the actual struct — do not hand-compute
or guess the layout.

## Example

`id` is at offset `0` and `read_count` at offset `4`; both fall in cache
line `0`, and pattern 1 writes both, so pattern 1 is false sharing
(`true`). `items[4]` sits near the end of the `items` array (offset `48`,
still cache line `0`), while `padding[1]` sits at offset `64` (cache line
`1`) — different lines, so pattern 5 is **not** false sharing (`false`)
even though both threads write.

## What the gate checks

The fixed driver (`main.cpp`) calls `classify_false_sharing()` and prints
the struct size followed by the 10 labels as `0`/`1`. The gate is an
exact string match (`exact_match == 1.0`) against the reference's printed
output — the reference computes its labels from real `offsetof` values
on the real compiled struct, not a hand-typed answer key, so every label
must match what the actual memory layout produces.
