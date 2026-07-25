#include <cstdio>
#include "sol.hpp"

// FIXED driver. 3 interleaved streams: stream 0 has a constant stride of
// 64 over 5 accesses; stream 1 has a constant stride of 64 over 3
// accesses; stream 2 has irregular deltas (100, then 200) over 3
// accesses and never confirms a pattern.
int main() {
    const int num_streams = 3;
    const int n = 11;
    int stream_id[n] = {0, 1, 0, 2, 1, 0, 2, 0, 1, 2, 0};
    long addr[n]      = {0, 1000, 64, 5000, 1064, 128, 5100, 192, 1128, 5300, 256};

    long count = stride_prefetch_count(stream_id, addr, n, num_streams);
    printf("prefetched=%ld\n", count);
    return 0;
}
