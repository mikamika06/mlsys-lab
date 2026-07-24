#include "sol.hpp"

// UNOPTIMIZED: numerically correct, but re-reads state->center.x/y/z on
// every iteration instead of hoisting them out of the loop. Fix this by
// loading center.x/y/z into locals once, before the loop.
void apply_shift(Point* pts, int n, const State* state) {
    for (int i = 0; i < n; i++) {
        double cx = mem_read_double(&state->center.x);
        double px = mem_read_double(&pts[i].x);
        mem_write_double(&pts[i].x, px + cx);

        double cy = mem_read_double(&state->center.y);
        double py = mem_read_double(&pts[i].y);
        mem_write_double(&pts[i].y, py + cy);

        double cz = mem_read_double(&state->center.z);
        double pz = mem_read_double(&pts[i].z);
        mem_write_double(&pts[i].z, pz + cz);
    }
}
