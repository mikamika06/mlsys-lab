#pragma once

// ============================================================================
// `n` loads (0..n-1), each taking the same fixed miss latency. `num_edges`
// dependency edges: edge i means load dep_to[i] cannot ISSUE until load
// dep_from[i] has COMPLETED (e.g. pointer chasing -- dep_to[i]'s address is
// computed from dep_from[i]'s result). The edges form a DAG (no cycles).
//
// Under ASAP scheduling with unlimited outstanding-miss buffers (issue a
// load the instant all its dependencies have completed), derive the
// MAXIMUM MEMORY-LEVEL PARALLELISM: the largest number of loads that are
// ever simultaneously in flight (issued but not yet completed) at once.
// ============================================================================
int mlp_degree(int n, const int* dep_from, const int* dep_to, int num_edges);
