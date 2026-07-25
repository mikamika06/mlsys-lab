// Reference: per-edge fusion-boundary decision. Cutting (materializing the
// intermediate to global memory) costs 2*size (one store by the producer,
// one load by each of the u consumers is amortized into the per-consumer
// cost model as a single reload -- the constant here is the write+read of
// the boundary itself). Fusing instead means every consumer beyond the
// first must recompute the producer's work: (reuse-1) extra recomputes at
// `recompute` cost each. Cut when cutting is cheaper or equal.
__global__ void fusion_boundary(const int* size, const int* reuse, const int* recompute, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int cut_cost = 2 * size[i];
        int fuse_cost = recompute[i] * (reuse[i] - 1);
        if (cut_cost <= fuse_cost) {
            out[i] = 1;
        } else {
            out[i] = 0;
        }
    }
}
