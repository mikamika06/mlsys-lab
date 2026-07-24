#include "sol.hpp"

// TODO: run the naive, 1-level-tiled, and 2-level-tiled i-j-k loop orders
// over the same matmul, each against its own reset_cache(), writing
// miss_count() after each into out[0], out[1], out[2]. See sol.hpp.
void matmul_miss_triple(int N, int tile1, int tile2,
                         long a_base, long b_base, long c_base, long* out) {
    (void)N; (void)tile1; (void)tile2;
    (void)a_base; (void)b_base; (void)c_base;
    // your code here
    out[0] = 0;
    out[1] = 0;
    out[2] = 0;
}
