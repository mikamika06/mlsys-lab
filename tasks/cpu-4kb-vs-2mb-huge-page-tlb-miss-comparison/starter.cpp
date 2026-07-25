#include "sol.hpp"

// TODO: replay the same base+i*stride access pattern (count elements,
// `passes` repeats) once against a TLB reset to 4096-byte pages and once
// against a TLB reset to 2*1024*1024-byte pages, writing miss_count()
// after each into out[0] and out[1]. See sol.hpp.
void tlb_miss_pair(long base, long stride, int count, int passes, long* out) {
    (void)base; (void)stride; (void)count; (void)passes;
    // your code here
    out[0] = 0;
    out[1] = 0;
}
