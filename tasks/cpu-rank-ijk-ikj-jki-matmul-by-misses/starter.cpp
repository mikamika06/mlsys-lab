#include "sol.hpp"

// TODO: run three passes of C += A*B over the same n x n x n index space
// -- one nested as i,j,k; one as i,k,j; one as j,k,i -- each on a fresh
// cache_reset(), touching a_addr/b_addr/c_addr in that order per (i,j,k).
// Read cache_misses() after each pass and write "ijk"/"ikj"/"jki" into
// out[0..3) sorted fewest-misses-first (ties: ijk < ikj < jki). See
// sol.hpp.
void rank_matmul_orders(int n, char out[3][4]) {
    (void)n;
    // your code here
    out[0][0] = '?'; out[0][1] = '\0';
    out[1][0] = '?'; out[1][1] = '\0';
    out[2][0] = '?'; out[2][1] = '\0';
}
