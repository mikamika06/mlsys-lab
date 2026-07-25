#include "sol.hpp"

double ridge_point(double peak_flops, double peak_bw) {
    return peak_flops / peak_bw;
}
