// BUG: re-derives (row, col) from tid the way real 2D matrix code would,
// then re-flattens with row and col SWAPPED (`col * h + row` instead of
// `row * w + col`) -- an accidental transpose. The array is still
// touched exactly once per element (correctness is unaffected: every
// valid index gets visited by exactly one thread), but consecutive
// threads land `h` elements apart instead of 1 apart, scattering a
// single warp's access across many far-apart memory segments.
__global__ void elementwise_scale(float* out, const float* in, int w, int h) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n = w * h;
    if (tid < n) {
        int row = tid / w;
        int col = tid % w;
        int idx = col * h + row;
        out[idx] = 2.0f * in[idx] + 1.0f;
    }
}
