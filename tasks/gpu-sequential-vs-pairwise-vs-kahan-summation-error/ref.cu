// Reference: Kahan compensated summation. `c` tracks the low-order bits
// that got rounded away on the previous addition and feeds them back in
// on the next one, instead of silently losing them.
__global__ void kahan_sum(float* out, const float* values, int n) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float s = 0.0f;
        float c = 0.0f;
        for (int i = 0; i < n; i++) {
            float y = values[i] - c;
            float t = s + y;
            c = (t - s) - y;
            s = t;
        }
        out[0] = s;
    }
}
