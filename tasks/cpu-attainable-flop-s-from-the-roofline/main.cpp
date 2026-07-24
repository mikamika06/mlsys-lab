#include <cstdio>
#include "sol.hpp"

// FIXED driver: five (peak_flops, peak_bandwidth, arithmetic_intensity)
// scenarios, spanning the compute-bound side, the memory-bound side, and
// the exact ridge point where both terms are equal.

static void run_case(double peak_flops, double peak_bandwidth, double ai) {
    double r = attainable_flops(peak_flops, peak_bandwidth, ai);
    printf("%.6f\n", r);
}

int main() {
    run_case(200.0, 50.0, 10.0);  // bandwidth term 500 > 200 -> compute-bound: 200
    run_case(200.0, 50.0, 2.0);   // bandwidth term 100 < 200 -> memory-bound: 100
    run_case(200.0, 50.0, 4.0);   // bandwidth term 200 == 200 -> ridge point: 200
    run_case(80.0, 40.0, 1.0);    // bandwidth term 40 < 80 -> memory-bound: 40
    run_case(80.0, 40.0, 3.0);    // bandwidth term 120 > 80 -> compute-bound: 80
    return 0;
}
