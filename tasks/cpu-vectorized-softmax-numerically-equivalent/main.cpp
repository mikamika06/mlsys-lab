#include <cstdio>
#include "sol.hpp"

// FIXED driver. A deliberately wide-range logits fixture: several values
// close together near the top, and both very large and very negative
// outliers -- exactly the kind of input that overflows a naive
// exp(logits[i]) without max-subtraction.
constexpr int N = 8;
constexpr float LOGITS[N] = {1000.0f, 999.0f, 998.5f, -50.0f, 0.0f, 2.0f, -1000.0f, 500.0f};

int main() {
    float probs[N];
    softmax(LOGITS, N, probs);
    for (int i = 0; i < N; ++i) {
        printf("p[%d]=%.6f\n", i, probs[i]);
    }
    return 0;
}
