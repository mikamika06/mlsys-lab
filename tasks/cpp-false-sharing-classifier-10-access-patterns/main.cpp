#include <cstdio>
#include "sol.hpp"

// FIXED driver. Do not edit. Calls the learner's classifier and prints the
// struct size plus the 10 labels as 0/1.
int main() {
    auto result = classify_false_sharing();
    const auto& labels = result.first;
    long struct_size = result.second;

    printf("struct_size=%ld\n", struct_size);
    for (bool b : labels) printf("%d ", b ? 1 : 0);
    printf("\n");
    return 0;
}
