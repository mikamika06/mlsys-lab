// BUG: fuse_cost forgets that the FIRST consumer never needs a recompute --
// only the other (reuse-1) consumers do. Charging `recompute * reuse`
// instead of `recompute * (reuse - 1)` overcounts the fusion cost by one
// full recompute on every edge, tipping some genuinely-fuse-worthy edges
// over into "cut" instead.
__global__ void fusion_boundary(const int* size, const int* reuse, const int* recompute, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int cut_cost = 2 * size[i];
        int fuse_cost = recompute[i] * reuse[i];
        if (cut_cost <= fuse_cost) {
            out[i] = 1;
        } else {
            out[i] = 0;
        }
    }
}
