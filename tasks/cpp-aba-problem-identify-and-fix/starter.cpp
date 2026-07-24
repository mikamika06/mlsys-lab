// Starter: ABA-vulnerable. It packs a tag field into the head, but never
// advances it, so the tag is stuck at 0 forever and the CAS effectively
// degenerates into a plain index compare — exactly the bug this task asks
// you to fix.
#include "sol.hpp"

PopCtx pop_begin(StackState& s) {
    uint64_t h = s.head.load();
    int idx = unpack_idx(h);
    PopCtx ctx{};
    ctx.expected_head = h;
    ctx.idx = idx;
    ctx.next_idx = (idx == NULL_IDX) ? NULL_IDX : s.pool[idx].next;
    return ctx;
}

PopResult pop_commit(StackState& s, const PopCtx& ctx) {
    if (ctx.idx == NULL_IDX) {
        return PopResult{NULL_IDX, false};
    }
    uint64_t expected = ctx.expected_head;
    // BUG: tag is never advanced (should be unpack_tag(ctx.expected_head) + 1).
    uint64_t desired = pack_head(ctx.next_idx, 0);
    if (s.head.compare_exchange_strong(expected, desired)) {
        return PopResult{ctx.idx, true};
    }
    return PopResult{NULL_IDX, false};
}

void push(StackState& s, int idx) {
    while (true) {
        uint64_t cur = s.head.load();
        int top_idx = unpack_idx(cur);
        s.pool[idx].next = top_idx;
        // BUG: tag is never advanced (should be unpack_tag(cur) + 1).
        uint64_t desired = pack_head(idx, 0);
        if (s.head.compare_exchange_strong(cur, desired)) {
            return;
        }
    }
}
