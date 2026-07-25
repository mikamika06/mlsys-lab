#include "sol.hpp"

void softmax(const float* logits, int n, float* probs) {
    // your code here
    (void)logits;
    for (int i = 0; i < n; ++i) probs[i] = 0.0f;
}
