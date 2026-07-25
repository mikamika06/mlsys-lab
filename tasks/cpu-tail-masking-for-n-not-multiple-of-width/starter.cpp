#include "sol.hpp"

// BUG: only runs the WIDTH-wide main loop. When n is not a multiple of
// WIDTH, the last n % WIDTH elements are never written.
void vec_add(const float* a, const float* b, float* c, int n) {
    int main_loop_end = (n / WIDTH) * WIDTH;
    for (int i = 0; i < main_loop_end; i += WIDTH) {
        for (int lane = 0; lane < WIDTH; ++lane) {
            c[i + lane] = a[i + lane] + b[i + lane];
        }
    }
}
