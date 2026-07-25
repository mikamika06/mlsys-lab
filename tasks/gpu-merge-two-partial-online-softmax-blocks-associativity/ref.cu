// Reference: merge two blocks' partial "safe softmax" statistics into one.
// Block k's stats (m_k, l_k) summarize a chunk of scores that was never
// materialized together with the other chunk: m_k is that chunk's own max,
// l_k = sum(exp(score - m_k)) over that chunk only. Merging them into the
// stats for the FULL sequence (both chunks together) needs to rescale each
// l_k from its own local max onto the new global max before adding:
//   m = max(m1, m2)
//   l = l1 * exp(m1 - m) + l2 * exp(m2 - m)
// This is exactly what a normal safe-softmax computed over the whole
// sequence at once would produce -- merging is associative.
__global__ void merge_online_softmax(float* m_out, float* l_out,
                                      const float* m1, const float* l1,
                                      const float* m2, const float* l2, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float m = fmaxf(m1[i], m2[i]);
        float l = l1[i] * expf(m1[i] - m) + l2[i] * expf(m2[i] - m);
        m_out[i] = m;
        l_out[i] = l;
    }
}
