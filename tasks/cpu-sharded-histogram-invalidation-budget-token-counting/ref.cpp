#include "sol.hpp"

// Pad each thread's block up to one full 64-byte cache line, even though
// only NUM_BINS * 8 = 32 bytes of it are live data -- guarantees no two
// threads' blocks ever share a line.
size_t thread_block_stride() { return 64; }
