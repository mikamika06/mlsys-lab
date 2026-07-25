#include "sol.hpp"
#include <cstdio>

// FIXED driver.
//
// Scenario 1: one attention+MLP layer, d_model=512, n_heads=8, d_head=64,
// seq_len=128, d_ff=4096. QKVO weights and Q/K/V/attn-output activations
// in fp16 (2 bytes), attention scores in fp32 (4 bytes, the usual
// softmax-stability choice), MLP weights and hidden activation in fp16.
// Cache levels 48 KiB / 1.5 MiB / 3 MiB -- deliberately small enough
// that the two big MLP weight matrices (4 MiB each) don't fit even the
// largest level, so they stream; everything else is small enough to be
// resident in level 1 (1.5 MiB). The MLP weights are modeled as each
// read twice (the layer processes the sequence in 2 chunks small enough
// to fit alongside the activations, so a non-resident weight has to be
// re-streamed from DRAM once per chunk).
//
// Scenario 2: a tiny layer where every tensor fits in the smallest
// cache level -- nothing streams, so the DRAM budget is just the sum of
// the tensor sizes, independent of num_uses.
int main() {
    {
        const int n = 12;
        static const long bytes[12] = {
            524288, 524288, 524288, 524288,   // W_Q, W_K, W_V, W_O (512*512*2)
            131072, 131072, 131072,           // Q_proj, K_proj, V_proj (128*512*2)
            524288,                           // attn_scores (128*128*8*4)
            131072,                           // attn_output (128*512*2)
            4194304, 4194304,                 // W_up, W_down (512*4096*2)
            1048576,                          // mlp_hidden (128*4096*2)
        };
        static const int uses[12] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1};
        static const long caps[3] = {49152, 1572864, 3145728};  // 48 KiB, 1.5 MiB, 3 MiB
        int residency[12];
        long budget = classify_layer_residency(bytes, uses, n, caps, 3, residency);
        printf("scenario=1 budget=%ld residency=", budget);
        for (int i = 0; i < n; i++) printf("%d ", residency[i]);
        printf("\n");
    }
    {
        const int n = 4;
        static const long bytes[4] = {4096, 8192, 2048, 16384};
        static const int uses[4] = {3, 5, 2, 4};
        static const long caps[2] = {32768, 262144};  // 32 KiB, 256 KiB
        int residency[4];
        long budget = classify_layer_residency(bytes, uses, n, caps, 2, residency);
        printf("scenario=2 budget=%ld residency=", budget);
        for (int i = 0; i < n; i++) printf("%d ", residency[i]);
        printf("\n");
    }
    return 0;
}
