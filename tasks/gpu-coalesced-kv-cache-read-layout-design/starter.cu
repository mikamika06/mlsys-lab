// Decode-step KV-cache read: gather K[:, :, t, :] (every layer, every
// head, one fixed sequence position) out of a cache stored
// [layers, heads, seq_len, dim] (dim fastest-varying, then seq, then
// head, then layer -- standard C-contiguous order).
//
// Design HOW consecutive thread ids map to (layer, head, d) triples so
// that consecutive threads read consecutive addresses in `kv` --
// whichever axis is fastest-varying in the physical layout (dim) must
// also be the fastest-varying axis across thread ids. Every (layer,
// head, d) triple must still be covered exactly once, and
// out[(layer*heads+head)*dim + d] must get kv's value at that triple's
// address -- but you choose which thread computes which triple.
__global__ void decode_read(float* out, const float* kv,
                             int layers, int heads, int seq_len, int dim, int t) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // your code here
}
