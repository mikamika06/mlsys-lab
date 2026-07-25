// LayerNorm backward: given dy (upstream gradient), x (forward input),
// and gamma (the scale parameter), compute dx, dgamma, dbeta. One
// thread per (row, col) element, blockDim.x = B*D. See task.md for the
// closed-form gradient formulas.
__global__ void layernorm_backward(float* dx, float* dgamma, float* dbeta, const float* dy, const float* x, const float* gamma, int B, int D) {
    // TODO
}
