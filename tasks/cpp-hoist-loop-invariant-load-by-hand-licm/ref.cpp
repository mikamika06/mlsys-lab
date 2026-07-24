#include "sol.hpp"

// Hoisted: center.x/y/z are each read exactly once, before the loop.
void apply_shift(Point* pts, int n, const State* state) {
    double cx = mem_read_double(&state->center.x);
    double cy = mem_read_double(&state->center.y);
    double cz = mem_read_double(&state->center.z);

    for (int i = 0; i < n; i++) {
        double px = mem_read_double(&pts[i].x);
        mem_write_double(&pts[i].x, px + cx);

        double py = mem_read_double(&pts[i].y);
        mem_write_double(&pts[i].y, py + cy);

        double pz = mem_read_double(&pts[i].z);
        mem_write_double(&pts[i].z, pz + cz);
    }
}
