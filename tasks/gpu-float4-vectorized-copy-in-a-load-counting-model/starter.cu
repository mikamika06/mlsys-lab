// Model a float4-vectorized copy of `in[0..n)` into `out`. Thread t (for
// t < n/4) handles group t: elements 4t, 4t+1, 4t+2, 4t+3, reported as ONE
// load instruction (`load_flag[t] = 1.0f`) even though this subset can
// only express it as 4 scalar accesses -- a real float4 load fetches all
// 4 in one instruction. The leftover `n % 4` elements each need their own
// thread doing one genuine scalar load (`load_flag[t] = 1.0f` for that
// thread too). Every other thread does nothing and reports 0.0f.
__global__ void vectorized_copy_load_count(float* out, float* load_flag,
                                            const float* in, int n) {
    int t = threadIdx.x;
    int num_groups = n / 4;
    int tail = n % 4;
    // TODO: implement the group copy + scalar tail + load_flag reporting
    // described above.
    load_flag[t] = 0.0f;
}
