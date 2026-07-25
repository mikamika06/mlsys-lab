// Sum values[0..n) with Kahan compensated summation: track the rounding
// error `c` lost on each addition and correct the next term for it,
// instead of a plain running total that silently swallows small terms
// added to a much larger accumulator.
__global__ void kahan_sum(float* out, const float* values, int n) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // your code here
    }
}
