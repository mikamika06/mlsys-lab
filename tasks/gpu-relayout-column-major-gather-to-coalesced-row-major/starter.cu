// Relayout an n x n matrix stored COLUMN-major (src[c*n+r] = matrix[r][c])
// into a ROW-major destination (dst[r*n+c] = matrix[r][c]).
//
// This version picks the thread mapping that makes the OUTPUT write
// coalesced (c fastest, matching dst's row-major layout) -- but that same
// mapping makes the SOURCE read a column-major "gather": consecutive
// threads jump n elements apart in src, one 128-byte transaction per lane
// instead of one per warp. Fix it by staging through __shared__ so both
// the read and the write can each use the mapping THEY want.
__global__ void relayout_col_to_row(float* dst, const float* src, int n) {
    int tid = threadIdx.x;
    int r = tid / n;
    int c = tid % n;
    dst[r * n + c] = src[c * n + r];
}
