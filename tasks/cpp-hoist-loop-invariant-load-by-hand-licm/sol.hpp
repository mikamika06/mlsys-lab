#pragma once

struct Point {
    double x, y, z;
};

// Under the LP64 ABI: int(4) + 4 bytes padding (Point needs 8-byte
// alignment) + Point(24) = 32 bytes total; `center` sits at offset 8.
struct State {
    int active;
    Point center;
};

// Instrumented memory access counters (harness code, defined in main.cpp).
// mem_read_double counts every double-precision load; mem_write_double is
// uncounted (this exercise is only about redundant LOADS).
extern int g_load_count;
double mem_read_double(const double* addr);
void mem_write_double(double* addr, double val);

// Shift each of the n Points at `pts` by state->center (x, y, z):
//     pts[i].x += state->center.x
//     pts[i].y += state->center.y
//     pts[i].z += state->center.z
//
// Access EVERY double through mem_read_double / mem_write_double (never a
// raw `->` / `.` read) so g_load_count reflects real memory traffic.
//
// state->center is loop-invariant: it never changes across iterations.
// You MUST hoist its three reads OUT of the loop — read center.x, center.y,
// center.z exactly once each, into locals, before the loop — and reuse
// those locals for every point, instead of re-reading center on every
// iteration.
void apply_shift(Point* pts, int n, const State* state);
