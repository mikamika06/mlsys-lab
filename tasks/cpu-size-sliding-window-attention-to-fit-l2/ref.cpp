#include "sol.hpp"

long attention_working_set_bytes(int W, int D, int elem_bytes, int score_bytes) {
    long fixed = 2L * D * elem_bytes;
    long per_w = 2L * D * elem_bytes + score_bytes;
    return fixed + (long)W * per_w;
}

int choose_max_window(long l2_capacity_bytes, int D, int elem_bytes, int score_bytes) {
    long fixed = 2L * D * elem_bytes;
    long per_w = 2L * D * elem_bytes + score_bytes;
    if (l2_capacity_bytes < fixed) return 0;
    long w = (l2_capacity_bytes - fixed) / per_w;
    return (int)w;
}
