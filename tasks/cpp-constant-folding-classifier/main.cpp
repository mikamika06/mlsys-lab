#include <cstdio>
#include "sol.hpp"

// FIXED driver. Do not edit. Calls the learner's classifier (which takes no
// arguments -- the 12 expressions are the fixed, documented set in
// sol.hpp) and prints whatever it returns as a space-separated line.
int main() {
    std::vector<int> r = classify_constant_folding();
    for (int v : r) printf("%d ", v);
    printf("\n");
    return 0;
}
