// Decode-step KV-cache read: gather K[:, :, t, :] (every layer, every
// head, one fixed sequence position) out of a cache stored
// [layers, heads, seq_len, dim] (dim fastest-varying, then seq, then
// head, then layer -- standard C-contiguous order).
//
// Design choice: DIM is the fastest-varying axis across consecutive
// thread ids, matching the fastest-varying axis in the physical layout
// -- so 32 consecutive threads (one warp, dim == 32) read 32 consecutive
// floats, exactly one 128-byte transaction.
__global__ void decode_read(float* out, const float* kv,
                             int layers, int heads, int seq_len, int dim, int t) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int d = tid % dim;
    int head = (tid / dim) % heads;
    int layer = tid / (dim * heads);

    int addr = ((layer * heads + head) * seq_len + t) * dim + d;
    int out_idx = (layer * heads + head) * dim + d;
    out[out_idx] = kv[addr];
}
