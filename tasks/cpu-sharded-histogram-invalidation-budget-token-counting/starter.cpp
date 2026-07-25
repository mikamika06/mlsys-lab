#include "sol.hpp"

// VALID but suboptimal: tightly pack each thread's block with no padding
// at all -- minimal memory, but every block sits right next to its
// neighbor, so half the threads share a cache line with another thread.
size_t thread_block_stride() { return NUM_BINS * sizeof(long long); }
