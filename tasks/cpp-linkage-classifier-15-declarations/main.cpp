#include <cstdio>
#include "sol.hpp"

// FIXED driver. Do not edit. Calls the learner's classifier and prints
// sizeof(S) plus the 15 linkage labels, one per line.
int main() {
    auto result = classify_linkage();
    const auto& labels = result.first;
    long struct_size = result.second;

    printf("struct_size=%ld\n", struct_size);
    for (const auto& lbl : labels) printf("%s\n", lbl.c_str());
    return 0;
}
