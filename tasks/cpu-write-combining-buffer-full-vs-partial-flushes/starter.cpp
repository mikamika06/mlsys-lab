#include "sol.hpp"

// TODO: track one WC entry per active line (FIFO slots), record byte
// offsets per line, flush immediately on completion (full_flush), evict
// the oldest entry when a new line needs a slot (full_flush if the
// evicted entry was complete, else partial_flush), and flush everything
// remaining at the end. See sol.hpp.
void wc_flush_stats(const long* addrs, int n, int line_bytes, int slots, long* out) {
    (void)addrs; (void)n; (void)line_bytes; (void)slots;
    // your code here
    out[0] = 0;
    out[1] = 0;
}
