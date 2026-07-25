#include <cstdio>
#include <map>
#include "sol.hpp"

// FIXED driver. Models a 16-thread binary-tree reduction: round r's
// active writers are every tid that's a multiple of 2^(r+1), for
// r = 0..3 (8, then 4, then 2, then 1 writer -- 15 writes total). No
// real threads and no timing (both non-deterministic) -- just a
// deterministic cache-coherence proxy: for each write in schedule
// order, compute which 64-byte line the slot falls in; if that line is
// currently "owned" (last written) by a DIFFERENT thread, that write is
// a false-sharing invalidation.
int main() {
    const int N = 16;
    int stride = 8 + slot_pad_bytes();

    std::map<long, int> owner;  // 64B line index -> owning thread id
    int invalidations = 0;

    for (int round = 0; round < 4; round++) {
        int step = 1 << (round + 1);
        for (int tid = 0; tid < N; tid += step) {
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
