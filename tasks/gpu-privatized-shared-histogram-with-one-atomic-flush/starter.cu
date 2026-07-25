// TODO: bin input[i] into a private per-block shared histogram of 8 bins
// (atomicAdd into shared memory, since threads in the same block may land
// on the same bin), then flush each bin once per block into the global
// histogram `out` with atomicAdd (since different blocks' flushes may
// target the same global bin). See task.md.
__global__ void histogram_privatized(const float* input, float* out, int n) {
    // your code here
}
