// Reference: relayout an n x n matrix stored COLUMN-major (src[c*n+r] =
// matrix[r][c]) into a ROW-major destination (dst[r*n+c] = matrix[r][c]),
// with BOTH the global read and the global write coalesced.
//
// A warp can only coalesce into few transactions if consecutive threads
// touch consecutive addresses. src and dst disagree about which of (r, c)
// should vary fastest -- src wants r fastest (column-major), dst wants c
// fastest (row-major) -- so no single thread-to-(r,c) mapping coalesces
// both directly. Stage through __shared__ instead: read src with the
// mapping THAT read wants (r fastest), write the tile with the mapping
// the OUTPUT wants (c fastest) after a barrier, then every thread's global
// write is also coalesced.
__global__ void relayout_col_to_row(float* dst, const float* src, int n) {
    __shared__ float tile[256];
    int tid = threadIdx.x;

    // Stage: r varies fastest across the warp -> src[c*n+r] is stride-1.
    int r = tid % n;
    int c = tid / n;
    tile[r * n + c] = src[c * n + r];
    __syncthreads();

    // Flush: c varies fastest across the warp -> dst[r*n+c] is stride-1.
    int r2 = tid / n;
    int c2 = tid % n;
    dst[r2 * n + c2] = tile[r2 * n + c2];
}
