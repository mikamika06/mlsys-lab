// Reference: n is not a multiple of blockDim.x * gridDim.x, so the last
// block has threads with i >= n. `out` is sized to the FULL launch
// (blockDim.x * gridDim.x), padded past n, and every thread stores to it
// unconditionally -- that part needs no mask. What needs masking is the
// LOAD: `in` only has n real elements, so reading in[i] for i >= n would
// run off the end of it. Mask the load itself (Triton's
// `tl.load(ptr, mask=mask, other=0.0)` pattern) so out-of-range threads
// contribute a defined 0.0 instead of whatever happens to sit past `in`.
__global__ void masked_scale_fill(float* out, const float* in, int n, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (i < n) ? in[i] : 0.0f;
    out[i] = s * v;
}
