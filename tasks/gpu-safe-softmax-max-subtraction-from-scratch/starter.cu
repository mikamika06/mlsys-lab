// Numerically-safe softmax, three single-threaded passes: find the row
// max m; sum exp(x[i] - m); emit out[i] = exp(x[i] - m) / sum. The max
// subtraction keeps every exponent <= 0 so exp() can't overflow.
__global__ void safe_softmax(float* out, const float* x, int n) {
    // TODO: guard threadIdx.x == 0. Pass 1: m = max over x[0..n). Pass
    // 2: s = sum of expf(x[i]-m). Pass 3: out[i] = expf(x[i]-m)/s.
}
