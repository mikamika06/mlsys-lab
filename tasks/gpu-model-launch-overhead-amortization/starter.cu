// Single thread. Derive N* = the smallest integer element count at which
// launch_cost / N <= per_elem_cost -- the point where a launch's fixed
// overhead has amortized down to no more than what one element's own
// compute already costs.
__global__ void crossover_n(float* out, float launch_cost, float per_elem_cost) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // your code here
    }
}
