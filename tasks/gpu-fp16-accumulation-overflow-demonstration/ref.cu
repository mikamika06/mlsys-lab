// Reference: sequentially accumulate x[0..n), saturating at clamp_max
// after every addition -- models the fact that a floating-point format
// can't represent anything past its own max finite magnitude, so
// exceeding it saturates (or overflows to infinity; saturating is the
// simpler, still-honest stand-in this simulator's plain float64
// arithmetic can express). Called with clamp_max = fp16's max finite
// value, the running sum saturates partway through; called with an
// effectively unbounded clamp_max, it never saturates at all.
__global__ void accumulate_clamped(const float* x, float* out, int n, float clamp_max) {
    float acc = 0.0;
    int i = 0;
    while (i < n) {
        acc = acc + x[i];
        if (acc > clamp_max) {
            acc = clamp_max;
        }
        i = i + 1;
    }
    out[0] = acc;
}
