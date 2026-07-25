#include "sol.hpp"

// TODO: maintain per-stream last address + last delta; on the 3rd+
// access of a stream, prefetch (and count) when this access's delta
// matches the stream's recorded delta. See sol.hpp.
long stride_prefetch_count(const int* stream_id, const long* addr, int n, int num_streams) {
    (void)stream_id; (void)addr; (void)n; (void)num_streams;
    // your code here
    return 0;
}
