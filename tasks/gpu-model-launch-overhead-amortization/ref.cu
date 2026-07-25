// Reference (single thread): given a fixed per-launch overhead and a
// per-element compute cost, derive the crossover element count N* at
// which the amortized overhead-per-element (launch_cost / N) drops to
// (or below) the per-element compute cost.
__global__ void crossover_n(float* out, float launch_cost, float per_elem_cost) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = ceilf(launch_cost / per_elem_cost);
    }
}
