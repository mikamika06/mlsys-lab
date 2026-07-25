// BUG: a one-stage software-pipelined loop -- each iteration prefetches
// chunk `c` into `next` while consuming the PREVIOUSLY prefetched chunk
// (`buf`, fetched last iteration) -- but the loop only ever consumes
// what it prefetched on an EARLIER iteration. When the loop exits,
// `buf` holds the very last chunk (num_chunks - 1) that was ever
// prefetched, and it is silently dropped: it's never added to `acc`.
// Fix it by draining that final buffered chunk after the loop.
__global__ void pipelined_sum(const float* x, float* out, int n, int num_chunks) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float acc = 0.0;
        float buf = x[i * num_chunks + 0];  // prologue: prefetch chunk 0
        int c = 1;
        while (c < num_chunks) {
            float next = x[i * num_chunks + c];  // prefetch chunk c
            acc = acc + buf;                      // consume the chunk fetched last iteration
            buf = next;
            c = c + 1;
        }
        out[i] = acc;
    }
}
