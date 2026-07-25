// Work-efficient exclusive scan (Blelloch), single block, n=32.
// Up-sweep (reduce) builds partial sums in place in a balanced binary
// tree over shared memory; the last element is cleared; down-sweep walks
// the same tree back down, turning it into an exclusive prefix sum.
__global__ void scan(float* out, const float* in, int n) {
    __shared__ float temp[32];
    int tid = threadIdx.x;
    temp[tid] = in[tid];
    __syncthreads();

    // Up-sweep (reduce) phase.
    int offset = 1;
    int d = n / 2;
    while (d > 0) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset = offset * 2;
        d = d / 2;
    }

    // Clear the last element -- root of the tree becomes the identity,
    // turning an inclusive reduce into an exclusive scan.
    if (tid == 0) {
        temp[n - 1] = 0;
    }

    // Down-sweep phase.
    d = 1;
    while (d < n) {
        offset = offset / 2;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
        d = d * 2;
    }

    __syncthreads();
    out[tid] = temp[tid];
}
