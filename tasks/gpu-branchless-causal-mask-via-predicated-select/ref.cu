// Branchless causal mask: EVERY thread always loads score[idx] and
// always stores out[idx] -- the mask decision is folded into the VALUE
// via a 0/1 predicate multiplied through, never into which memory ops a
// thread issues. Every thread in a warp performs the same access
// sequence (one load, one store) regardless of the mask outcome.
__global__ void causal_mask(float* out, const float* score, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx / n;
    int j = idx % n;
    float v = score[idx];
    float keep = (j <= i) ? 1.0f : 0.0f;
    float neg_inf = -1.0e30f;
    out[idx] = keep * v + (1.0f - keep) * neg_inf;
}
