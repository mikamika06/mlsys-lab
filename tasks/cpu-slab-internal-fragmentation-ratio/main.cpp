#include <cstdio>
#include "sol.hpp"

// FIXED driver. Size classes 16/32/64/128/256; 6 requests spanning tiny
// (1 byte, worst case) to exact-fit.
int main() {
    const int num_classes = 5;
    int size_classes[num_classes] = {16, 32, 64, 128, 256};
    const int n = 6;
    int requests[n] = {1, 17, 32, 100, 129, 256};

    double ratio = slab_fragmentation_ratio(size_classes, num_classes, requests, n);
    printf("avg_ratio=%.6f\n", ratio);
    return 0;
}
