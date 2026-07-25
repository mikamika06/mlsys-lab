// Reference: Hillis-Steele inclusive scan over 8 elements, one thread
// per element. At each doubling step, every thread first READS the
// value `offset` slots below itself into a local register (`val`),
// then a barrier, THEN writes `sdata[tid] += val` behind a second
// barrier -- the read is fully separated from every thread's write by
// a __syncthreads(), so no thread can ever read a neighbor's
// already-updated (this-step) value instead of last step's.
__global__ void inclusive_scan(float* out, const float* in, int n) {
    int tid = threadIdx.x;
    __shared__ float sdata[8];
    sdata[tid] = in[tid];
    __syncthreads();

    int offset = 1;
    while (offset < n) {
        float val = 0.0;
        if (tid >= offset) {
            val = sdata[tid - offset];
        }
        __syncthreads();
        if (tid >= offset) {
            sdata[tid] = sdata[tid] + val;
        }
        __syncthreads();
        offset = offset * 2;
    }

    out[tid] = sdata[tid];
}
