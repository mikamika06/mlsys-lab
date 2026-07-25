// Merge two blocks' partial "safe softmax" statistics (m1[i], l1[i]) and
// (m2[i], l2[i]) -- each computed over a DIFFERENT chunk of the same row's
// scores, using that chunk's own local max -- into the statistics for the
// row's FULL sequence (m_out[i], l_out[i]).
__global__ void merge_online_softmax(float* m_out, float* l_out,
                                      const float* m1, const float* l1,
                                      const float* m2, const float* l2, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // TODO: m_out[i] = max(m1[i], m2[i]); l_out[i] = the two partial
        // sums rescaled onto that new max and added together.
    }
}
