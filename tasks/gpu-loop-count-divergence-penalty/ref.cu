// Reference (single thread): a warp's 32 lanes each ran a data-dependent
// loop trips[lane] times. Because a warp executes in lockstep, EVERY
// lane's hardware slot has to keep cycling until the SLOWEST lane
// finishes -- lanes that are already done sit predicated-off, still
// occupying an issue slot every remaining iteration. Serialized
// iterations = max(trips); wasted lane-iterations = 32*max - sum(trips)
// (every lane-iteration beyond what that lane actually needed).
__global__ void divergence_penalty(float* out, const float* trips, int warp_size) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float mx = trips[0];
        float sum = trips[0];
        for (int i = 1; i < warp_size; i++) {
            if (trips[i] > mx) {
                mx = trips[i];
            }
            sum = sum + trips[i];
        }
        out[0] = mx;
        out[1] = warp_size * mx - sum;
    }
}
