// Geometry-independent dropout mask: the per-element RNG depends only on
// the flat element index i (never on thread/block shape), so the mask is
// identical for any launch geometry that covers all n elements.
//
// This CUDA-C subset has no bitwise operators, so the hash uses only
// +, -, *, /, %: an explicit `% 4294967296` (2**32) stands in for 32-bit
// unsigned wraparound, and `/ 16777216` (2**24) followed by `% 256` stands
// in for extracting a byte the way `>> 24 & 0xff` would on real hardware.
__global__ void dropout_mask_kernel(float* mask, int seed, int n, int limit) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int h = (seed + i * 2654435761) % 4294967296;
        int r = (h / 16777216) % 256;
        if (r < limit) {
            mask[i] = 1.0;
        } else {
            mask[i] = 0.0;
        }
    }
}
