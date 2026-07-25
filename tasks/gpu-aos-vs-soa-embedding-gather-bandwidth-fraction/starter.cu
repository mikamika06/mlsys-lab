// Gather each thread's embedding row from an SoA (dimension-major) table
// and sum its D dimensions into out[i]. `emb` is laid out emb[d*V + v]:
// dimension d, vocab row v. Thread i must read row v = idx[i].
// Keep the per-step access across the warp coalesced: at loop step d,
// every thread's address must be d*V + idx[i] (consecutive across i),
// never idx[i]*D + d.
__global__ void gather_soa(float* out, const float* emb, const int* idx, int D, int V) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: v = idx[i]; sum emb[d*V + v] for d in [0, D); out[i] = sum;
}
