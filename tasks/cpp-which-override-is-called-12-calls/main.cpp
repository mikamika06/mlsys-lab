#include <cstdio>
#include "sol.hpp"

// FIXED driver. Do not edit. It asks the solution for its predicted 12-tag
// vector and prints it. The grader compares this printed vector, exactly,
// against the reference (which computes the tags via real virtual dispatch).
int main() {
    int out[12];
    for (int i = 0; i < 12; i++) out[i] = -1;  // deterministic sentinel
    predict_tags(out);
    for (int i = 0; i < 12; i++) printf("%d ", out[i]);
    printf("\n");
    return 0;
}
