#pragma once

// Pinned NUMA model constants (defined in main.cpp, do not redefine):
//   LOCAL_NS           average latency (ns) of an access served by the
//                       local node's memory controller.
//   REMOTE_NS           average latency (ns) of an access served by a
//                       remote node's controller over the interconnect.
//   CONTENTION_COEF_NS  queueing penalty (ns) added per unit of
//                       "effective concurrent sharer" beyond one -- see
//                       classify_workload below.
extern const double LOCAL_NS;
extern const double REMOTE_NS;
extern const double CONTENTION_COEF_NS;

// One workload: num_nodes NUMA nodes run threads that all touch one
// shared array. access_count[k] (k in [0, num_nodes)) is how many of
// the workload's total memory accesses are issued by node k's threads.
struct Workload {
    int num_nodes;
    const long* access_count;
};

// LEARNER IMPLEMENTS.
//
// Compute both placement policies' average memory access time (ns) for
// `w`, write them to *bind_ns / *interleave_ns, and return which policy
// wins: the exact lowercase string "interleave" or "bind".
//
// BIND (numactl --membind to a single node): place every page of the
// shared array on whichever ONE node minimizes AMAT. Since a local
// access is always cheaper than a remote one, that is always the node
// with the highest access_count -- call its share of the total
// L = max(access_count) / total (sum of access_count over all nodes):
//
//   base_bind_ns = L * LOCAL_NS + (1 - L) * REMOTE_NS
//
// But once every page lives on that one node, its memory controller
// must field every node's requests (not just its own), so it queues
// under concurrent cross-node pressure. Measure how many nodes are
// *effectively* sharing the resource with the standard concentration
// measure (the inverse Herfindahl index) over each node's share
// f_k = access_count[k] / total:
//
//   effective_sharers = 1 / sum_k(f_k^2)          (ranges [1, num_nodes];
//                                                    1 = one node owns it
//                                                    all, num_nodes = every
//                                                    node's share is equal)
//   bind_ns = base_bind_ns + CONTENTION_COEF_NS * (effective_sharers - 1)
//
// INTERLEAVE (round-robin every page across all num_nodes nodes): any
// access, regardless of which node issued it, lands local with
// probability 1/num_nodes and remote otherwise -- and because the
// array's pages (and therefore the load) are already spread evenly
// across every node's controller by construction, there is no added
// queueing term:
//
//   interleave_ns = LOCAL_NS / num_nodes
//                  + REMOTE_NS * (num_nodes - 1) / num_nodes
//
// Return "interleave" iff interleave_ns < bind_ns, else "bind" (bind
// wins ties, since it never underperforms a truly uncontended local
// workload).
const char* classify_workload(const Workload& w, double* bind_ns, double* interleave_ns);
