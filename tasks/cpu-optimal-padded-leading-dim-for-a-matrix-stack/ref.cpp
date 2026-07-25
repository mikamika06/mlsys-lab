#include "sol.hpp"

int choose_padded_ld(int n_cols, int M, int elem_bytes, int line_bytes, int num_sets) {
    long period = (long)line_bytes * (long)num_sets;
    for (int ld = n_cols; ; ld++) {
        long stride = (long)M * (long)ld * (long)elem_bytes;
        if (stride % period != 0) return ld;
    }
}
