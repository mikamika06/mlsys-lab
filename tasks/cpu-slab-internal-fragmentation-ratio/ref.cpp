#include "sol.hpp"

double slab_fragmentation_ratio(const int* size_classes, int num_classes, const int* requests, int n) {
    double sum_ratio = 0.0;
    for (int i = 0; i < n; i++) {
        int r = requests[i];
        int allocated = size_classes[num_classes - 1];
        for (int c = 0; c < num_classes; c++) {
            if (size_classes[c] >= r) {
                allocated = size_classes[c];
                break;
            }
        }
        sum_ratio += (double)allocated / (double)r;
    }
    return sum_ratio / n;
}
