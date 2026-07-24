#pragma once

// Deterministic ILP critical-path model (harness code, defined in
// main.cpp). A loop that accumulates into a SINGLE running sum is a
// serial dependency chain: step i cannot start until step i-1's result
// is ready, so an n-element reduction costs n cycles no matter how many
// execution ports the CPU has. Splitting the accumulation across several
// INDEPENDENT running sums breaks that chain into several shorter,
// mutually-independent ones that the CPU can execute in parallel; the
// critical path becomes the length of the LONGEST individual chain, not
// their total op count.
//
// report_op(chain_id): call once for every per-element fused
// multiply-add, tagging which independent accumulator chain (0-based) it
// belongs to.
// critical_path_cycles(): the number of report_op() calls made for
// whichever chain_id was reported the most times since the last
// reset_ilp() -- i.e. the longest chain's length.
void reset_ilp();
void report_op(int chain_id);
long critical_path_cycles();

// dot_product_ilp: compute sum_i a[i]*b[i] for i in [0, n) using
// `num_chains` independent running-sum accumulators, round-robin
// assigned: element i's multiply-add accumulates into chain
// (i % num_chains). Call report_op(i % num_chains) exactly once per
// element (n calls total). After the loop, sum the num_chains partial
// accumulators together (that final combine is NOT reported -- adding a
// handful of partials is negligible next to the n-element loop) and
// return the total.
double dot_product_ilp(const double* a, const double* b, int n, int num_chains);
