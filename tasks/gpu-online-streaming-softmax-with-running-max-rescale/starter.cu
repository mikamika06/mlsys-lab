// Online (streaming) softmax over x[0..n). First pass, single-threaded,
// starting from m=x[0], l=1: for each subsequent x[i], m_new =
// max(m,x[i]); l = l*exp(m-m_new) + exp(x[i]-m_new); m = m_new. Second
// pass: out[i] = exp(x[i]-m)/l for every i, using the FINAL m and l.
__global__ void online_softmax(float* out, const float* x, int n) {
    // TODO: guard threadIdx.x == 0. Run the online max+rescale pass
    // (see sol comment above), then the normalization pass.
}
