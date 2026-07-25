#pragma once

// LEARNER IMPLEMENTS.
//
// A pointer-chasing loop that prefetches `distance` hops ahead issues
// one new prefetch every `cycles_per_iter` cycles of work; each
// prefetched load takes `latency_cycles` cycles to complete. By
// Little's Law, fully hiding that latency needs enough prefetches in
// flight at once to cover it:
//
//   required_concurrency = ceil(latency_cycles / cycles_per_iter)
//
// But real hardware can only track a fixed number of outstanding memory
// requests at once (its miss-status-holding registers): `mlp_max`. No
// matter how far ahead software prefetches, it can never usefully keep
// more than `mlp_max` requests in flight at the same time.
//
// Return the MINIMUM prefetch distance (in loop iterations) that
// saturates whichever of those two limits binds first:
//
//   distance = min(required_concurrency, mlp_max)
int min_saturating_distance(int latency_cycles, int cycles_per_iter, int mlp_max);

// Whether, at that minimum saturating distance, the loop is fully
// latency-hidden (true) or is instead capped by the hardware's MLP
// limit and still exposes some latency on every iteration (false):
// true iff required_concurrency <= mlp_max.
bool is_latency_fully_hidden(int latency_cycles, int cycles_per_iter, int mlp_max);
