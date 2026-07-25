#include "sol.hpp"

// TODO: starting at ld = n_cols, increase ld until
// (M * ld * elem_bytes) % (line_bytes * num_sets) != 0, and return that ld.
int choose_padded_ld(int n_cols, int M, int elem_bytes, int line_bytes, int num_sets) {
    (void)M; (void)elem_bytes; (void)line_bytes; (void)num_sets;
    // your code here
    return n_cols;
}
