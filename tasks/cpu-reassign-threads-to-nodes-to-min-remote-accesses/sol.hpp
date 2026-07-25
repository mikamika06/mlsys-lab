#pragma once

// LEARNER IMPLEMENTS.
//
// `T` threads must each be pinned to exactly one of `N` NUMA nodes, with
// exactly `capacity` threads assigned per node (the caller guarantees
// T == N * capacity, so every node fills exactly). `access_count[t*N+n]`
// is how many memory accesses thread `t` makes to data that already
// lives on node `n` (data placement is fixed; only which node each
// thread RUNS ON is up for choice). Pinning thread `t` to node `n` makes
// exactly `access_count[t*N+n]` of its accesses local; every other
// access it makes -- to data that lives on some other node -- is
// remote.
//
// Search over every assignment of threads to nodes that respects the
// per-node capacity, and return the MINIMUM total number of remote
// accesses (summed over all T threads) achievable over any such
// assignment. A per-thread greedy pick (send each thread to its own
// best node, first-come-first-served on capacity) is NOT guaranteed to
// find this minimum: two threads can both want the same crowded node,
// and the right answer is to displace whichever of them loses the LEAST
// by moving to its next-best node -- which a naive greedy that only
// checks "is there room" won't necessarily identify.
long min_remote_accesses(int T, int N, int capacity, const long* access_count);
