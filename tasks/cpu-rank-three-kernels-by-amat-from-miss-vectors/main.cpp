#include <cstdio>
#include "sol.hpp"

// FIXED driver. 3 kernels with deliberately non-monotonic miss vectors
// (the kernel with the most L1 misses is NOT the one with the worst
// AMAT, since its L2 behavior differs).
int main() {
    long accesses[3]  = {1000, 1000, 1000};
    long l1_misses[3] = {200, 50, 500};
    long l2_misses[3] = {50, 40, 10};

    double amat[3] = {-1.0, -1.0, -1.0};  // sentinel
    int rank[3] = {-1, -1, -1};

    rank_by_amat(accesses, l1_misses, l2_misses, amat, rank);

    printf("amat0=%.4f amat1=%.4f amat2=%.4f rank=%d,%d,%d\n",
           amat[0], amat[1], amat[2], rank[0], rank[1], rank[2]);
    return 0;
}
