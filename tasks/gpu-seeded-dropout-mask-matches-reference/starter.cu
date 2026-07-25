// dropout_mask: for each i in [0, n), decide keep (1.0f) or drop (0.0f)
// deterministically from `seed` and i alone -- no state carried between
// elements, no RNG builtin. Hash h = (seed + i*2654435761) mod 1000000007
// (both large constants are just fixed, arbitrary numbers -- nothing
// about them is special beyond being fixed), then u = h / 1000000007.0f
// is a pseudo-uniform value in [0, 1). Drop (mask[i] = 0.0f) if u < p,
// otherwise keep (mask[i] = 1.0f).
__global__ void dropout_mask(float* mask, int n, int seed, float p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // TODO: compute h, then u, then mask[i] as described above.
    }
}
