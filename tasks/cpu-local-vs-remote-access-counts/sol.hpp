#pragma once

// ============================================================================
// LEARNER implements this in solve.cpp.
//
// A NUMA machine partitions physical memory into fixed `page_bytes`-byte
// pages, each pinned to exactly one NUMA node. `node_of_page[q]` gives the
// node that owns page `q`, for `q` in `[0, num_pages)`. A thread pinned to
// `home_node` runs an access trace of `n` byte addresses `addrs[0..n)`,
// with `0 <= addrs[i] < num_pages * page_bytes` for every `i`.
//
// For each access: its page is `addrs[i] / page_bytes`, and that page's
// node is `node_of_page[page]`. The access is LOCAL if that node equals
// `home_node`, REMOTE otherwise (a different node -- reached over the
// interconnect, at higher latency). Write the number of local accesses to
// `out[0]` and the number of remote accesses to `out[1]`
// (`out[0] + out[1] == n` always).
// ============================================================================
void count_local_remote(const long* addrs, int n, long page_bytes,
                         const int* node_of_page, int num_pages,
                         int home_node, long* out);
