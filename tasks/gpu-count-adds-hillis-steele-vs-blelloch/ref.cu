// Two inclusive-scan algorithms over the same n=32 input, single block.
//
// hillis_steele_scan: the textbook parallel scan. Every one of the n
// threads is active at every one of the log2(n) steps -- doing O(n log n)
// total work, more than the O(n) a sequential scan needs.
//
// blelloch_scan: the work-efficient scan. Up-sweep (reduce) then
// down-sweep, each phase touching a SHRINKING (up-sweep) or GROWING
// (down-sweep) number of active threads -- O(n) total work.

__global__ void hillis_steele_scan(float* out, const float* in, int n) {
    __shared__ float temp[512];  // double buffer: temp[0..n) and temp[n..2n)
    int tid = threadIdx.x;
    int pout = 0;
    int pin = 1;
    temp[pout * n + tid] = in[tid];
    __syncthreads();

    int offset = 1;
    while (offset < n) {
        pout = 1 - pout;
        pin = 1 - pin;
        if (tid >= offset) {
            temp[pout * n + tid] = temp[pin * n + tid] + temp[pin * n + tid - offset];
        } else {
            temp[pout * n + tid] = temp[pin * n + tid];
        }
        __syncthreads();
        offset = offset * 2;
    }
    out[tid] = temp[pout * n + tid];
}

__global__ void blelloch_scan(float* out, const float* in, int n) {
    __shared__ float temp[256];
    int tid = threadIdx.x;
    temp[tid] = in[tid];
    __syncthreads();

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

    if (tid == 0) {
        temp[n - 1] = 0;
    }

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
    out[tid] = temp[tid] + in[tid];  // exclusive scan -> inclusive scan
}
