#include <cstdio>
#include <map>
#include "sol.hpp"

// FIXED driver. Models 8 threads doing 5 ROUNDS of round-robin counter
// increments each (a much heavier, repeated-contention stress test than
// a single pass): thread 0 writes, then thread 1, ..., then thread 7,
// repeated 5 times -- 40 writes total. No real threads and no timing
// (both non-deterministic) -- just a deterministic cache-coherence
// proxy: for each write in order, compute which 64-byte line the
// counter falls in; if that line is currently "owned" (last written) by
// a DIFFERENT thread, that write is a false-sharing invalidation.
int main() {
    const int N = 8, ROUNDS = 5;
    int stride = 8 + counter_pad_bytes();

    std::map<long, int> owner;  // 64B line index -> owning thread id
    int invalidations = 0;

    for (int r = 0; r < ROUNDS; r++) {
        for (int tid = 0; tid < N; tid++) {
            long addr = (long)tid * stride;
            long line = addr / 64;
            auto it = owner.find(line);
            if (it != owner.end() && it->second != tid) {
                invalidations++;
            }
            owner[line] = tid;
        }
    }

    printf("stride=%d invalidations=%d\n", stride, invalidations);
    return 0;
}
