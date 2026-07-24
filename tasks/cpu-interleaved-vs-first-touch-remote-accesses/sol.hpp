#pragma once

// Modeling convention: memory is divided into 4096-byte pages. There
// are `num_nodes` NUMA nodes; thread `t` always runs pinned to node
// `t % num_nodes`.
constexpr int PAGE_BYTES = 4096;

// One access in the trace: thread `thread` reads/writes byte address
// `addr`.
struct Access {
    int thread;
    long addr;
};

// For the SAME trace, count how many of its accesses are "remote"
// (the accessing thread's node differs from the page's home node)
// under two different page-placement policies:
//
//   FIRST-TOUCH: a page has no home node until some thread accesses it
//   for the very first time (in trace order) -- at that moment its
//   home node becomes the FIRST TOUCHING THREAD's node, permanently,
//   for every access to that page for the rest of the trace (including
//   that very first access, which is always local to itself).
//
//   INTERLEAVED: every page's home node is fixed in advance, before
//   the trace runs, independent of who touches it: page number `p`
//   (== addr / PAGE_BYTES) belongs to node `p % num_nodes`.
//
// Write the two totals (summed over all n accesses) into
// *first_touch_remote and *interleaved_remote.
void count_remote_accesses(const Access* trace, int n, int num_nodes,
                            long* first_touch_remote, long* interleaved_remote);
