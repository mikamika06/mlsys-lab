#include "sol.hpp"
#include <algorithm>

double compute_numa_amat(double l3_hit_time_ns, double l3_miss_rate,
                          double local_dram_latency_ns, double remote_base_latency_ns,
                          int num_remote_hops, double per_hop_latency_ns,
                          double local_dram_fraction) {
    double remote_latency = remote_base_latency_ns +
        (double)std::max(num_remote_hops - 1, 0) * per_hop_latency_ns;
    double dram_latency = local_dram_fraction * local_dram_latency_ns +
        (1.0 - local_dram_fraction) * remote_latency;
    return l3_hit_time_ns + l3_miss_rate * dram_latency;
}
