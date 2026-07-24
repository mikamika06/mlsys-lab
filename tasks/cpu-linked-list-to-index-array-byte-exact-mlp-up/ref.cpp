#include "sol.hpp"

void pointer_chase_traversal(const int* next_idx, int head, int n, int* order_out) {
    int cur = head;
    for (int i = 0; i < n; i++) {
        order_out[i] = cur;
        report_load(false);
        cur = next_idx[cur];
    }
}

void gather_by_index(const double* values, const int* order, int n, double* out) {
    for (int i = 0; i < n; i++) {
        out[i] = values[order[i]];
        report_load(true);
    }
}
