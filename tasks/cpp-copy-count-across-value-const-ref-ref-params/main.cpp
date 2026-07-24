#include <cstdio>
#include "sol.hpp"

// FIXED driver + fixed overloads. None of the three functions below does
// anything with `p` — the copy count is decided entirely by how the CALLER
// passes the argument, which is exactly what solve.cpp has to get right.

void process_value(Probe p) { (void)p; }
void process_const_ref(const Probe& p) { (void)p; }
void process_ref(Probe& p) { (void)p; }

int main() {
    int out[4] = {-1, -1, -1, -1};
    run_scenarios(out);

    printf("%d %d %d %d\n", out[0], out[1], out[2], out[3]);
    printf("sizeof(Probe)=%d\n", static_cast<int>(sizeof(Probe)));
    return 0;
}
