#pragma once
// Contract for a tagged-pointer lock-free stack over a fixed node pool.
//
// The stack top is a single std::atomic<uint64_t> `head` that packs a
// 32-bit pool index (low bits) and a 32-bit ABA version tag (high bits).
// Nodes are never freed; a "popped" node is simply not reachable from
// `head` until something pushes it again, so the classic ABA scenario
// (pop, pop, push-back-the-same-node) can happen without any real
// allocator involved.
//
// pop is split into two phases so the driver in main.cpp can script an
// exact interleaving deterministically (no real OS threads, no timing):
//   pop_begin  -> snapshot head + the top node's `next` (read-only)
//   pop_commit -> exactly one CAS attempt against that snapshot
//
// YOUR JOB: make pop_commit's CAS reject a snapshot whose tag is stale
// even when the index coincidentally matches again (the ABA case), and
// make push (and every successful pop_commit) advance the tag so a
// stale snapshot can never spuriously match later.

#include <atomic>
#include <cstdint>

constexpr int POOL_SIZE = 3;
constexpr int NULL_IDX = -1;

struct Node {
    int value;
    int next; // index into StackState::pool, or NULL_IDX
};

struct StackState {
    Node pool[POOL_SIZE];
    std::atomic<uint64_t> head; // packed (idx, tag) via pack_head/unpack_*
};

// Bit-packing helpers (not part of what you fix — pure utilities).
inline uint64_t pack_head(int idx, uint32_t tag) {
    uint32_t uidx = static_cast<uint32_t>(idx);
    return (static_cast<uint64_t>(tag) << 32) | uidx;
}
inline int unpack_idx(uint64_t h) {
    return static_cast<int>(static_cast<uint32_t>(h & 0xFFFFFFFFu));
}
inline uint32_t unpack_tag(uint64_t h) {
    return static_cast<uint32_t>(h >> 32);
}

struct PopCtx {
    int idx;               // node observed at the top, or NULL_IDX if empty
    int next_idx;          // pool[idx].next observed at the same moment
    uint64_t expected_head; // exact head snapshot pop_begin observed
};

struct PopResult {
    int idx;     // popped node index, or NULL_IDX on failure/empty
    bool success;
};

// Phase 1 (read-only): snapshot the current head and, if non-empty, the
// next pointer of the top node. Must not modify `s` in any way.
PopCtx pop_begin(StackState& s);

// Phase 2 (commit): perform exactly ONE compare-and-swap attempt of
// s.head from ctx.expected_head to pack_head(ctx.next_idx, <new tag>).
// The CAS must fail (leaving `s` unchanged) whenever s.head no longer
// equals ctx.expected_head — including when only the tag changed while
// the idx happens to match again. On success return {ctx.idx, true};
// on failure, or if ctx.idx == NULL_IDX, return {NULL_IDX, false}.
PopResult pop_commit(StackState& s, const PopCtx& ctx);

// Push pool[idx] onto the stack. Must retry its own CAS loop until it
// succeeds, and MUST advance the tag on every successful CAS (including
// this one) so that no PopCtx captured before this call can spuriously
// match a head value produced after it.
void push(StackState& s, int idx);
