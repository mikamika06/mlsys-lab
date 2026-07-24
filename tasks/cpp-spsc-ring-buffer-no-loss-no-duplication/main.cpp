#include <cstdio>
#include <cstdint>
#include <vector>
#include "sol.hpp"

// FIXED deterministic driver. It exercises the SPSC ring buffer single-threaded
// and prints numbers that a correct implementation reproduces exactly and a
// wrong/empty one does not.

int main() {
    // ---- Part A: direct full/empty semantics on a tiny buffer -------------
    // Capacity 4, push 6 -> exactly 4 must be accepted, 2 rejected (full).
    RingBuffer a(4);
    int pushes_ok = 0;
    for (int i = 0; i < 6; ++i)
        pushes_ok += a.try_push(100 + i) ? 1 : 0;   // expect 4

    int sz_full = (int)a.size();                     // expect 4

    int32_t first = -1;
    int pop1 = a.try_pop(&first) ? 1 : 0;            // expect 1, first == 100
    int push_after_pop = a.try_push(200) ? 1 : 0;    // expect 1 (slot freed)

    printf("A %d %d %d %d\n", pushes_ok, sz_full, (int)first, push_after_pop);
    printf("A_reopen %d\n", pop1);

    // ---- Part B: interleaved stream, no loss / no duplication -------------
    // Eager producer (pushes every tick), throttled consumer (~40% of ticks),
    // so the buffer repeatedly fills to capacity and wraps around. A correct
    // FIFO delivers the produced sequence 0..M-1 exactly and in order.
    const int CAP = 4;
    const int M   = 40;
    RingBuffer q(CAP);

    std::vector<int32_t> received;
    received.reserve(M);
    int produced = 0;
    uint32_t rng = 2463534242u;                      // fixed seed
    long guard = 0;

    while ((produced < M || q.size() > 0) && guard++ < 500000) {
        rng = rng * 1664525u + 1013904223u;
        bool do_pop = ((rng >> 13) % 5u) < 2u;       // ~40% of ticks

        if (produced < M && q.try_push(produced)) ++produced;

        if (do_pop) {
            int32_t v = -1;
            if (q.try_pop(&v)) received.push_back(v);
        }
    }

    int n = (int)received.size();
    long sum = 0;
    int in_order = (n == M) ? 1 : 0;
    for (int i = 0; i < n; ++i) {
        sum += received[i];
        if (i >= M || received[i] != i) in_order = 0;
    }

    printf("B %d %d %ld\n", n, in_order, sum);
    printf("SEQ");
    for (int i = 0; i < n; ++i) printf(" %d", (int)received[i]);
    printf("\n");
    return 0;
}
