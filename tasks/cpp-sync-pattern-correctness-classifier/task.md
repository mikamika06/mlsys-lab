## Context

Concurrent C++ programs use mutexes, atomics, condition variables, and memory barriers to control access to shared state. A **data race** is two conflicting accesses (at least one a write) to the same non-atomic name with no synchronization between them. A **deadlock** is a cycle in the lock-order graph — thread A holds lock X and waits for Y while thread B holds Y and waits for X.

This task classifies each snippet from a fixed *abstract description* (which names it reads, writes, and treats atomically; which lock pairs and lock-order edges it has) rather than by executing real threads — the same names, in the same combinations, always classify the same way.

## Task

`sol.hpp` declares `SyncPattern` (vectors of `reads`, `writes`, `atomics` names, plus `locks` and `lock_edges` as `(string, string)` pairs). Implement:

```cpp
void classify_sync_patterns(const SyncPattern* patterns, int n, std::string* out);
```

For each pattern, apply these checks **in order** and stop at the first match:

1. If any pair in `locks` has equal elements (`a == b`) → `"deadlock"`.
2. Build a directed graph from `lock_edges` (`held -> requested`). If it contains a cycle reachable from any node with an outgoing edge → `"deadlock"`.
3. Let `shared` = (`reads` ∪ `writes`) − `atomics`. If any name is in both `writes` and `reads`, and it's in `shared` (i.e. not atomic) → `"data-race"`.
4. Otherwise → `"ok"`.

## Example

A pattern with `writes = {"value"}`, `reads = {"value"}`, and no `atomics` writes and reads the same non-atomic name — that's `"data-race"`. A pattern with `lock_edges = {(a,b), (b,c), (c,a)}` has a 3-cycle in its lock-order graph — that's `"deadlock"`, even though nothing in `reads`/`writes` looks unsafe.

## What the gate checks

`main.cpp` builds ten fixed `SyncPattern`s (mirroring real-world shapes: a plain atomic counter, an unsynchronized shared variable, a two-lock ABBA deadlock, a self-lock, an atomic flag guarding a non-atomic payload, a three-lock cycle, and more) and prints each classification. The candidate's full stdout is compared byte-for-byte (`exact_match = 1.0`) against the reference's. Checking the shared-access rule before the lock-cycle rule, or forgetting that atomics are excluded from the race check, both mislabel at least one of the ten cases.
