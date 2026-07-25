#include "sol.hpp"

// TODO: derive AMAT = l3_hit_time_ns + l3_miss_rate * dram_latency, where
// dram_latency mixes local and remote DRAM latency by local_dram_fraction,
// and remote latency adds a per-hop cost beyond the first hop. See sol.hpp.
double compute_numa_amat(double l3_hit_time_ns, double l3_miss_rate,
                          double local_dram_latency_ns, double remote_base_latency_ns,
                          int num_remote_hops, double per_hop_latency_ns,
                          double local_dram_fraction) {
    (void)l3_hit_time_ns; (void)l3_miss_rate;
    (void)local_dram_latency_ns; (void)remote_base_latency_ns;
    (void)num_remote_hops; (void)per_hop_latency_ns; (void)local_dram_fraction;
    // your code here
    return 0.0;
}
