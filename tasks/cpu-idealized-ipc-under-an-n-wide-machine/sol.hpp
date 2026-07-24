#pragma once
#include <vector>

// ============================================================================
// LEARNER implements this in solve.cpp.
//
// idealized_ipc: given `n = latency.size()` instructions where instruction i
// has execution latency `latency[i]` cycles, and `deps[i]` lists the indices
// of instructions that must FINISH before instruction i may START (a DAG —
// deps[i] never contains i or anything that (transitively) depends on i),
// compute the idealized IPC (instructions retired per cycle) of a
// `width`-wide superscalar machine with perfect scheduling: no structural
// hazards, no branch mispredicts, no port contention — the only limits are
// (a) data dependencies and (b) how many instructions can issue per cycle.
//
// Two independent lower bounds on the number of cycles this DAG can
// possibly finish in:
//   - critical_path: the length, in cycles, of the longest dependency
//     chain. finish[i] = start[i] + latency[i], where start[i] is 0 if
//     deps[i] is empty, else max(finish[d] for d in deps[i]). The critical
//     path is max(finish[i]) over all i.
//   - width_bound: n / (double)width — even with zero dependencies, a
//     width-wide machine cannot retire more than `width` instructions in
//     any one cycle.
//
// The idealized number of cycles is whichever bound is larger:
//     ideal_cycles = max(critical_path, width_bound)
// and the idealized IPC is:
//     idealized_ipc = n / ideal_cycles
// ============================================================================
double idealized_ipc(const std::vector<int>& latency,
                      const std::vector<std::vector<int>>& deps,
                      int width);
