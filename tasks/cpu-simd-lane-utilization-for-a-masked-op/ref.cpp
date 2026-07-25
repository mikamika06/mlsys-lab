#include "sol.hpp"

double simd_lane_utilization(const bool* mask, int n, int width) {
    int total_active = 0;
    int groups_executed = 0;
    for (int g = 0; g * width < n; g++) {
        int active_in_group = 0;
        for (int lane = 0; lane < width; lane++) {
            if (mask[g * width + lane]) active_in_group++;
        }
        total_active += active_in_group;
        if (active_in_group > 0) groups_executed++;
    }
    return (double)total_active / (double)(groups_executed * width);
}
