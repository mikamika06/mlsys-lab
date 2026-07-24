// Fixed driver: scripts a classic ABA interleaving deterministically
// (no real threads, no timing) and prints the resulting stack state as
// plain numbers so the candidate implementation can be compared against
// the reference byte-for-byte.
#include "sol.hpp"
#include <cstdio>

int main() {
    StackState s{};
    // Initial stack (top -> bottom): idx0(100) -> idx1(200) -> idx2(300) -> null
    s.pool[0] = Node{100, 1};
    s.pool[1] = Node{200, 2};
    s.pool[2] = Node{300, NULL_IDX};
    s.head.store(pack_head(0, 0));

    // "Thread A" begins a pop, decides to remove idx0, then stalls before
    // committing its CAS (this is the classic ABA setup).
    PopCtx ctxA = pop_begin(s);

    // "Thread B" runs to completion in the meantime: pops idx0, pops idx1,
    // then pushes idx0 back on top. idx0 is now reused with the same index.
    PopCtx    ctxB1 = pop_begin(s);
    PopResult resB1 = pop_commit(s, ctxB1);

    PopCtx    ctxB2 = pop_begin(s);
    PopResult resB2 = pop_commit(s, ctxB2);

    push(s, 0);

    // Thread A resumes and tries to commit its now-stale snapshot.
    PopResult resA1 = pop_commit(s, ctxA);
    int retried = 0;
    PopResult resAfinal = resA1;
    if (!resA1.success) {
        // A correctly-tagged stack must reject the stale CAS; A retries
        // exactly once more with a fresh snapshot.
        retried = 1;
        PopCtx ctxA2 = pop_begin(s);
        resAfinal = pop_commit(s, ctxA2);
    }

    // Walk the final stack from head, cycle-guarded.
    uint64_t h = s.head.load();
    int cur = unpack_idx(h);
    int seq[POOL_SIZE + 1];
    int n = 0;
    bool visited[POOL_SIZE] = {false, false, false};
    while (cur != NULL_IDX) {
        if (n > POOL_SIZE || cur < 0 || cur >= POOL_SIZE || visited[cur]) {
            n = -1; // corruption: cycle or out-of-range index
            break;
        }
        visited[cur] = true;
        seq[n++] = s.pool[cur].value;
        cur = s.pool[cur].next;
    }

    printf("resA1_success=%d\n", resA1.success ? 1 : 0);
    printf("retried=%d\n", retried);
    printf("resAfinal_success=%d\n", resAfinal.success ? 1 : 0);
    printf("resAfinal_value=%d\n", resAfinal.success ? s.pool[resAfinal.idx].value : -1);
    printf("poppedB1_value=%d\n", resB1.success ? s.pool[resB1.idx].value : -1);
    printf("poppedB2_value=%d\n", resB2.success ? s.pool[resB2.idx].value : -1);
    printf("stack_len=%d\n", n);
    printf("stack_values=");
    for (int i = 0; i < n; i++) {
        printf("%d%s", seq[i], (i + 1 < n) ? "," : "");
    }
    printf("\n");
    return 0;
}
