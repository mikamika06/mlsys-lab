#include <cstdio>
#include "sol.hpp"

// FIXED driver, two scenarios. For each: pick the max window, then print
// the working set at that window (must fit) and at one window larger
// (must NOT fit) -- proving the chosen window is the true maximum, not
// just "a" fitting window.
namespace {
void run_scenario(long l2_capacity_bytes, int D, int elem_bytes, int score_bytes) {
    int w = choose_max_window(l2_capacity_bytes, D, elem_bytes, score_bytes);
    long ws_at_w = attention_working_set_bytes(w, D, elem_bytes, score_bytes);
    long ws_at_w_plus_1 = attention_working_set_bytes(w + 1, D, elem_bytes, score_bytes);
    printf("L2=%ld D=%d elem_bytes=%d score_bytes=%d -> W=%d ws(W)=%ld ws(W+1)=%ld\n",
           l2_capacity_bytes, D, elem_bytes, score_bytes, w, ws_at_w, ws_at_w_plus_1);
}
}  // namespace

int main() {
    // fp32 K/V and scores, 256KB L2, head_dim 128.
    run_scenario(262144, 128, 4, 4);
    // bf16 K/V (2 bytes) but scores kept in fp32, 32KB L2, head_dim 64.
    run_scenario(32768, 64, 2, 4);
    return 0;
}
