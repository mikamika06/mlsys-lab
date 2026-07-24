#include "sol.hpp"

// TODO: for each access, determine its home node under BOTH policies
// (first-touch: whichever thread touches that page first, permanently;
// interleaved: page % num_nodes, fixed in advance) and count how many
// of the n accesses are remote under each. See sol.hpp.
void count_remote_accesses(const Access* trace, int n, int num_nodes,
                            long* first_touch_remote, long* interleaved_remote) {
    (void)trace; (void)n; (void)num_nodes;
    // your code here
    *first_touch_remote = 0;
    *interleaved_remote = 0;
}
