// TODO: sequentially accumulate x[0..n) into acc, but after EVERY
// addition, if acc has exceeded clamp_max, saturate it back down to
// clamp_max (a format can't hold anything past its own max finite
// value). See ref.cu's approach.
__global__ void accumulate_clamped(const float* x, float* out, int n, float clamp_max) {
    out[0] = 0.0;
}
