#include "sol.hpp"
#include <cmath>

void softmax(const float* logits, int n, float* probs) {
    float m = logits[0];
    for (int i = 1; i < n; ++i) {
        if (logits[i] > m) m = logits[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        probs[i] = std::exp(logits[i] - m);
        sum += probs[i];
    }

    for (int i = 0; i < n; ++i) {
        probs[i] = probs[i] / sum;
    }
}
