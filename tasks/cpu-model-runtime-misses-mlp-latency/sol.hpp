#pragma once

// Model the total memory-stall cycles of a loop whose cache misses are
// serviced by a non-blocking cache with memory-level parallelism (MLP):
// up to `mlp` independent misses can be outstanding at once, each taking
// `miss_latency` cycles. The modeled critical-path time is the number of
// "waves" of misses (num_misses spread across mlp concurrent slots)
// times the latency of a single miss:
//
//   modeled_cycles = (double)num_misses / (double)mlp * miss_latency
//
// A model that instead serializes every miss one at a time
// (num_misses * miss_latency) throws away all credit for overlap and
// badly overestimates runtime whenever mlp > 1 -- which is precisely
// the point of MLP: independent misses (distinct array elements) can be
// serviced concurrently, unlike true pointer-chasing where each load
// depends on the result of the last one and mlp is stuck at 1.
double modeled_cycles(long num_misses, int mlp, double miss_latency);
