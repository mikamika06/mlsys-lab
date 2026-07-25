#include "sol.hpp"

long classify_layer_residency(const long* tensor_bytes, const int* num_uses, int num_tensors,
                               const long* cache_capacities, int num_levels, int* residency_out) {
    long total = 0;
    for (int i = 0; i < num_tensors; i++) {
        int level = -1;
        for (int L = 0; L < num_levels; L++) {
            if (tensor_bytes[i] <= cache_capacities[L]) { level = L; break; }
        }
        residency_out[i] = level;
        if (level != -1) {
            total += tensor_bytes[i];
        } else {
            total += tensor_bytes[i] * (long)num_uses[i];
        }
    }
    return total;
}
