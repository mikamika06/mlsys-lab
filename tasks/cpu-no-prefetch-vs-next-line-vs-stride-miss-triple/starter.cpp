#include "sol.hpp"

// TODO: build address_k = base + k*stride_bytes for k in [0, n_steps)
// and feed each one, in order, through touch_no_prefetch/touch_next_line
// /touch_stride, then read the 3 miss counts into out[0..2]. See
// sol.hpp.
void generate_and_run(long base, int stride_bytes, int n_steps, long* out) {
    (void)base; (void)stride_bytes; (void)n_steps; (void)out;
    // your code here
}
