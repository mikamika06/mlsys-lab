#include "sol.hpp"

// TODO: recursively transpose the full N x N matrix in cache-oblivious
// fashion, on REAL memory: out[col*N+row] = in[row*N+col] for every
// (row, col), plus touch(in_addr(N,row,col)) / touch(out_addr(N,col,row))
// for each. See sol.hpp for the exact recursive contract (quadrant split
// on (r0, c0, n), n<=8 base case).
void co_transpose(const float* in, float* out, int N) {
    (void)in; (void)out; (void)N;
    // your code here
}
