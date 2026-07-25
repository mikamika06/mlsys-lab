// TODO: online-softmax attention forward, one thread per query row i.
// Maintain running m (max score so far, init -1e30), l (rescaled sum
// of exp(score-m), init 0), and acc0..acc3 (rescaled weighted sum of
// V, init 0). For each key j: score = dot(Q[i], K[j]) * scale;
// new_m = max(m, score); correction = exp(m - new_m); p =
// exp(score - new_m); l = l*correction + p; acc_d = acc_d*correction +
// p*V[j][d] for each d; m = new_m. After the loop, O[i][d] = acc_d / l.
// See ref.cu.
__global__ void flash_attention_fwd(const float* Q, const float* K, const float* V,
                                     float* O, int N, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        O[i * 4 + 0] = 0.0;
        O[i * 4 + 1] = 0.0;
        O[i * 4 + 2] = 0.0;
        O[i * 4 + 3] = 0.0;
    }
}
