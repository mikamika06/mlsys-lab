#include "sol.hpp"

// TODO: attention_working_set_bytes = 2*D*elem_bytes + W*(2*D*elem_bytes +
// score_bytes). choose_max_window = largest W>=0 with that <=
// l2_capacity_bytes (floor((l2_capacity_bytes - 2*D*elem_bytes) / (2*D*elem_bytes
// + score_bytes)), clamped to 0).
long attention_working_set_bytes(int W, int D, int elem_bytes, int score_bytes) {
    (void)W; (void)D; (void)elem_bytes; (void)score_bytes;
    // your code here
    return 0;
}

int choose_max_window(long l2_capacity_bytes, int D, int elem_bytes, int score_bytes) {
    (void)l2_capacity_bytes; (void)D; (void)elem_bytes; (void)score_bytes;
    // your code here
    return 0;
}
