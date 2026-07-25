// Reference: Volkov's latency-hiding model. A warp that issues a
// memory request stalls for `mem_latency` cycles unless something else
// is ready to run in the meantime. The scheduler can cover that stall
// with independent work from OTHER resident warps (occupancy) OR from
// the SAME warp's own independent, already-issued instructions
// (instruction-level parallelism, ILP) -- either source adds to the
// total outstanding concurrency. `concurrency = warps_resident * ilp`;
// whatever part of `mem_latency` concurrency doesn't cover is `exposed`
// stall time added on top of the kernel's own compute cycles.
__global__ void latency_hiding_cycles(int warps_resident, int ilp, int compute_cycles,
                                       int mem_latency, float* out) {
    float concurrency = warps_resident * ilp + 0.0;
    float exposed = mem_latency - concurrency;
    if (exposed < 0.0) {
        exposed = 0.0;
    }
    out[0] = compute_cycles + exposed;
}
