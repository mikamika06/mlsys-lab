// Compute the inclusive prefix sum of `in[0..n)` into `out[0..n)`
// (out[i] = in[0] + in[1] + ... + in[i]) using the Hillis-Steele
// log-step scan: one thread per element, `log2(n)` doubling steps.
// See task.md.
__global__ void inclusive_scan(float* out, const float* in, int n) {
    // TODO
}
