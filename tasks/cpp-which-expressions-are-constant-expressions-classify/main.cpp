#include <cstdio>
#include "sol.hpp"

// FIXED driver. Do not edit. Calls the learner's classifier, prints the 12
// bits (expression 1 -> bit 0, ... expression 12 -> bit 11), then the 12-bit
// mask value and the number of expressions marked as constant.
int main() {
    unsigned m = classify_constexpr();
    int pop = 0;
    for (int i = 0; i < 12; i++) {
        int b = (m >> i) & 1u;
        pop += b;
        printf("%d ", b);
    }
    printf("\nmask=%u popcount=%d\n", m & 0xFFFu, pop);
    return 0;
}
