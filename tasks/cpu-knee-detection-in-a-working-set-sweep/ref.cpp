#include "sol.hpp"

int detect_knees(const double* latency, int n, double rel_threshold, int* out_indices) {
    int count = 0;
    for (int i = 1; i < n; i++) {
        double prev = latency[i - 1];
        double rel = (latency[i] - prev) / prev;
        if (rel > rel_threshold) {
            out_indices[count++] = i;
        }
    }
    return count;
}
