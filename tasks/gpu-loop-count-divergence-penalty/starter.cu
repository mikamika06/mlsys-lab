// Single thread. `trips[0..warp_size)` are one warp's 32 per-lane loop
// trip counts. Compute:
//   out[0] = serialized iterations the warp actually takes = max(trips)
//   out[1] = wasted lane-iterations = warp_size * max(trips) - sum(trips)
__global__ void divergence_penalty(float* out, const float* trips, int warp_size) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // your code here
    }
}
