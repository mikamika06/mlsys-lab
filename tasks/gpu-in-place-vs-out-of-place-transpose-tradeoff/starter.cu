// Two transpose strategies for the SAME 16x16 matrix, in one file.
//
// transpose_in_place: transpose A in place -- ONLY A and n, no second
// buffer. Only let each mirrored pair of elements (row, col) with
// row < col swap once; leave the diagonal and the lower triangle alone
// (they get updated as a side effect of their upper-triangle partner's
// swap).
//
// transpose_out_of_place: write the transpose of `in` into `out`, leaving
// `in` unmodified.
__global__ void transpose_in_place(float* A, int n) {
    int tid = threadIdx.x;
    int row = tid / n;
    int col = tid % n;
    // TODO: if row < col, swap A[row*n+col] and A[col*n+row].
}

__global__ void transpose_out_of_place(float* out, const float* in, int n) {
    int tid = threadIdx.x;
    int row = tid / n;
    int col = tid % n;
    // TODO: out[col*n+row] = in[row*n+col];
}
