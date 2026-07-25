// Reference: modeled atomic-op count for warp-aggregated atomics. For
// warp i, `active_count[i]` threads out of its 32 lanes want to increment
// a shared counter. The naive strategy has every one of them issue its
// own atomicAdd(1): `active_count[i]` atomics. Warp aggregation instead
// has the warp vote (ballot) on who wants to increment, elects ONE leader
// lane, and that lane alone does a SINGLE atomicAdd(popcount) carrying the
// whole warp's contribution at once -- exactly one atomic per warp, but
// ONLY if at least one lane in it actually wants to increment (an entirely
// idle warp does zero atomics either way).
__global__ void warp_aggregated_atomic_counts(float* naive_out, float* warp_agg_out,
                                               const float* active_count, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        float n = active_count[i];
        naive_out[i] = n;
        warp_agg_out[i] = (n > 0.5f) ? 1.0f : 0.0f;
    }
}
