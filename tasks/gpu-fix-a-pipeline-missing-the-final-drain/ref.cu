// Fixed: same one-stage software-pipelined loop (prefetch the NEXT
// chunk while consuming the PREVIOUSLY prefetched one), plus the
// epilogue that was missing: after the loop exits, `buf` still holds
// the very last chunk that was ever prefetched -- it was never folded
// into `acc` inside the loop, so it has to be drained explicitly.
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
        acc = acc + buf;  // epilogue: drain the last prefetched chunk
        out[i] = acc;
    }
}
