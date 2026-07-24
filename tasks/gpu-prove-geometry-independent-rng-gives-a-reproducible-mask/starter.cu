// Compute the geometry-independent element index
// i = blockIdx.x * blockDim.x + threadIdx.x, guard i < n, then the
// per-element hash (only +, -, *, /, % are available — no bitwise ops):
//   h = (seed + i * 2654435761) % 4294967296;
//   r = (h / 16777216) % 256;
// then mask[i] = 1.0 if r < limit else 0.0.
__global__ void dropout_mask_kernel(float* mask, int seed, int n, int limit) {
    // your code here
}
