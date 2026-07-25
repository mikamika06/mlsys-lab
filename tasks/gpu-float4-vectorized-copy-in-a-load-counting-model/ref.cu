// Reference: model a float4-vectorized copy. This CUDA-C subset has no
// real float4 type, so the "vectorized load" is modeled as: one thread
// handles 4 CONSECUTIVE elements with 4 scalar accesses but reports ONE
// load instruction for the group (a real float4 load fetches all 4 in a
// single instruction); n is not a multiple of 4, so the leftover
// `n % 4` elements each need their own thread doing a genuine scalar
// load (one load instruction each). Threads beyond what's needed do
// nothing and report 0.
__global__ void vectorized_copy_load_count(float* out, float* load_flag,
                                            const float* in, int n) {
    int t = threadIdx.x;
    int num_groups = n / 4;
    int tail = n % 4;

    if (t < num_groups) {
        int base = t * 4;
        out[base + 0] = in[base + 0];
        out[base + 1] = in[base + 1];
        out[base + 2] = in[base + 2];
        out[base + 3] = in[base + 3];
        load_flag[t] = 1.0f;
    } else if (t < num_groups + tail) {
        int idx = num_groups * 4 + (t - num_groups);
        out[idx] = in[idx];
        load_flag[t] = 1.0f;
    } else {
        load_flag[t] = 0.0f;
    }
}
