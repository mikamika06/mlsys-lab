#pragma once
// Modeled Average Memory Access Time (ns) for a single-level cache backed
// by NUMA-aware DRAM.
//
//   remote_latency = remote_base_latency_ns
//                     + max(num_remote_hops - 1, 0) * per_hop_latency_ns
//   dram_latency   = local_dram_fraction * local_dram_latency_ns
//                     + (1 - local_dram_fraction) * remote_latency
//   AMAT           = l3_hit_time_ns + l3_miss_rate * dram_latency
//
// l3_hit_time_ns is the cache-hit time (t_cache), l3_miss_rate is the
// fraction of accesses that miss L3 (r), local_dram_fraction is the
// fraction of DRAM accesses served by the local NUMA node (f), and
// num_remote_hops (h, >= 1) is the hop count to the remote node.
double compute_numa_amat(double l3_hit_time_ns, double l3_miss_rate,
                          double local_dram_latency_ns, double remote_base_latency_ns,
                          int num_remote_hops, double per_hop_latency_ns,
                          double local_dram_fraction);
