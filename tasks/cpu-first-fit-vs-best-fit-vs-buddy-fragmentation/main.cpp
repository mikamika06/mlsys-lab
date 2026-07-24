#include <cstdio>
#include "sol.hpp"

// FIXED driver: 4 equal 40-byte allocs to fill most of the arena, free two
// of them leaving two separate holes, a 70-byte alloc that only fits in
// the untouched remainder, a 20-byte alloc that forces first-fit and
// best-fit to pick DIFFERENT holes, then one more free that coalesces
// differently for each policy.
int main() {
    int op_kind[9] = {0, 0, 0, 0, 1, 1, 0, 0, 1};
    int op_arg[9]  = {40, 40, 40, 40, 0, 2, 70, 20, 1};
    const int num_ops = 9;

    int out[3] = {-1, -1, -1};  // sentinel: an empty starter leaves this untouched
    fragmentation_after_trace(op_kind, op_arg, num_ops, out);

    printf("first_fit=%d best_fit=%d buddy=%d\n", out[0], out[1], out[2]);
    return 0;
}
