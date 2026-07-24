## Context

In lock-free programming, the **ABA problem** happens when a thread reads a value $A$ from a shared location, another thread changes it to $B$ and back to $A$, and the first thread's compare-and-swap (CAS) succeeds because the value *looks* unchanged — even though the underlying structure was modified in between. In a lock-free stack over a node pool, this can resurrect a node that was already removed, corrupting the structure without ever triggering a crash.

The standard fix is a **tagged pointer**: the head is not just an index, it is $(\text{index}, \text{version\_tag})$, and the tag is incremented on every successful CAS. Two heads with the same index but different tags are treated as different values, so a stale CAS is rejected instead of silently succeeding.

## Task

`sol.hpp` declares a fixed-size node pool `StackState` and a head packed as `pack_head(idx, tag) -> uint64_t` (index in the low 32 bits, tag in the high 32 bits). You must implement three functions:

- `pop_begin(s)` — read-only snapshot of the current head and the top node's `next`.
- `pop_commit(s, ctx)` — exactly one CAS attempt against that snapshot.
- `push(s, idx)` — push `pool[idx]`, retrying its own CAS loop until it succeeds.

The starter (`solve.cpp`) already packs a tag into the head, but never advances it — the tag is stuck at $0$ forever, so `compare_exchange_strong` degenerates into a plain index comparison. Fix `pop_commit` and `push` so every successful CAS advances the tag (`unpack_tag(old_head) + 1`), making a stale snapshot with a matching index but an old tag correctly rejected.

## Example

`main.cpp` scripts a fixed interleaving (no real OS threads, so it is fully deterministic): a "thread A" begins a pop of node `idx0` and stalls before committing. Meanwhile "thread B" pops `idx0`, pops `idx1`, then pushes `idx0` back — reusing the same index while thread A's stale snapshot is still around. Thread A then resumes and tries to commit.

With a correct tag, A's stale CAS is rejected (index matches, tag doesn't) and A must re-read and retry, ending with a stack that correctly lost only the node thread B never pushed back. With the tag stuck at $0$, A's stale CAS wrongly succeeds, and the node thread B already popped and kept reappears on the stack — an element simultaneously "held" by thread B and reachable from `head`.

## What the gate checks

`main.cpp` prints the outcome of thread A's first CAS attempt, whether it had to retry, the final popped values, and the resulting stack contents, all as plain integers. The candidate's full stdout is compared byte-for-byte (`exact_match = 1.0`) against the reference's stdout for the same scripted interleaving. A tag that never advances produces a different (corrupted) trace, so it fails the gate even though it never crashes or triggers a sanitizer.
