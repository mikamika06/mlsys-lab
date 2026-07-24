#include "sol.hpp"

double attainable_flops(double peak_flops, double peak_bandwidth, double arithmetic_intensity) {
    double bandwidth_term = arithmetic_intensity * peak_bandwidth;
    return bandwidth_term < peak_flops ? bandwidth_term : peak_flops;
}
