// Compute out[i,:] = LayerNorm(x[i,:] + residual[i,:]) * gamma + beta for
// N rows of D features, one thread per row (i = threadIdx.x). FUSE the
// residual add into the LayerNorm pass -- never write x+residual to a
// separate array in `out` or anywhere else and read it back; recompute
// x[i*D+d] + residual[i*D+d] directly wherever it's needed. See task.md
// for the exact two-pass structure (sum/sum-of-squares, then normalize).
__global__ void fused_layernorm_residual(float* out, const float* x, const float* residual,
                                          const float* gamma, const float* beta,
                                          int N, int D, float eps) {
    int i = threadIdx.x;
    // TODO: pass 1 accumulate sum/sumsq of x+residual; compute
    // mean/var/inv_std; pass 2 recompute x+residual, normalize, scale by
    // gamma, shift by beta, write to out.
}
