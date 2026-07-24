#pragma once

// Deterministic memory-level-parallelism (MLP) model (harness code,
// defined in main.cpp). The simulated core can have at most
// MAX_INFLIGHT=8 loads outstanding at once.
//
//   report_load(false)  -- a DEPENDENT load: its address depends on the
//     VALUE of a previous load, so it cannot even be issued until that
//     previous load has completed. Extends the single serial chain by 1.
//   report_load(true)   -- an INDEPENDENT load: its address is already
//     known (does not depend on any other load's result), so it is
//     issued into one of the MAX_INFLIGHT outstanding-load slots,
//     round-robin. That slot's chain extends by 1; different slots run
//     concurrently.
//
// critical_path_cycles() returns the length of the longest chain (serial
// chain, or busiest slot) since the last reset_mlp().
void reset_mlp();
void report_load(bool independent);
long critical_path_cycles();

// pointer_chase_traversal: follow the linked list starting at node
// `head` (next_idx[i] gives node i's successor, or -1 past the last
// node), writing the n-long visiting order into order_out[0..n). You
// cannot know node i+1's index until next_idx[current node] has
// actually been read -- this is inherently serial. Call
// report_load(false) exactly once per step (n calls total).
void pointer_chase_traversal(const int* next_idx, int head, int n, int* order_out);

// gather_by_index: given the traversal order computed above
// (order[0..n)), gather values[order[i]] into out[i] for every i. Every
// gather address is already known from order[] alone -- independent of
// any other gather's result -- so call report_load(true) exactly once
// per gather (n calls total) to get credit for the available
// memory-level parallelism.
void gather_by_index(const double* values, const int* order, int n, double* out);
