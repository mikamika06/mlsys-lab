#include "sol.hpp"

void vec_add(const float* a, const float* b, float* c, int n) {
    int i = 0;
    int main_loop_end = (n / WIDTH) * WIDTH;
    for (; i < main_loop_end; i += WIDTH) {
        for (int lane = 0; lane < WIDTH; ++lane) {
            c[i + lane] = a[i + lane] + b[i + lane];
        }
    }
    // Tail: the remaining n % WIDTH elements that don't fill a full
    // WIDTH-wide chunk.
    for (; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}
