// For warp i, `active_count[i]` of its 32 lanes want to increment a
// shared counter. Model the atomic-op count for both strategies:
//   naive_out[i]:     one atomicAdd per active lane -> active_count[i].
//   warp_agg_out[i]:  one atomicAdd for the WHOLE warp, but only if it
//                     has at least one active lane -> 1.0f if
//                     active_count[i] > 0, else 0.0f (an idle warp does
//                     no atomics under either strategy).
__global__ void warp_aggregated_atomic_counts(float* naive_out, float* warp_agg_out,
                                               const float* active_count, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        // TODO: naive_out[i] = active_count[i];
        // TODO: warp_agg_out[i] = (active_count[i] > 0.5f) ? 1.0f : 0.0f;
    }
}
