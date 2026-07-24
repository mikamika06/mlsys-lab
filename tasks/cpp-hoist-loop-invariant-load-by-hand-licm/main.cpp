#include <cstdio>
#include "sol.hpp"

int g_load_count = 0;

double mem_read_double(const double* addr) {
    g_load_count++;
    return *addr;
}

void mem_write_double(double* addr, double val) {
    *addr = val;
}

int main() {
    const int N = 5;
    Point pts[N];
    for (int i = 0; i < N; i++) {
        pts[i].x = i * 1.0;
        pts[i].y = i * 2.0 + 1.0;
        pts[i].z = i * 3.0 - 1.0;
    }
    State state;
    state.active = 1;
    state.center = {10.0, -20.0, 30.5};

    g_load_count = 0;
    apply_shift(pts, N, &state);

    printf("loads=%d\n", g_load_count);
    for (int i = 0; i < N; i++)
        printf("%.3f %.3f %.3f\n", pts[i].x, pts[i].y, pts[i].z);
    return 0;
}
