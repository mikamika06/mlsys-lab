// Reference: tagged-pointer CAS, ABA-safe.
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
    uint32_t new_tag = unpack_tag(ctx.expected_head) + 1;
    uint64_t desired = pack_head(ctx.next_idx, new_tag);
    // compare_exchange checks the FULL packed value, tag included, so a
    // stale snapshot whose idx matches again but whose tag doesn't is
    // correctly rejected.
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
        uint32_t new_tag = unpack_tag(cur) + 1;
        uint64_t desired = pack_head(idx, new_tag);
        if (s.head.compare_exchange_strong(cur, desired)) {
            return;
        }
        // else: retry with a fresh snapshot
    }
}
