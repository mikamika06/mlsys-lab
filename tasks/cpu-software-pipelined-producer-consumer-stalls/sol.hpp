#pragma once

struct StallCounts {
    long long naive_stalls;
    long long pipelined_stalls;
};

// ============================================================================
// LEARNER implements this in solve.cpp.
//
// Model an N-iteration producer/consumer loop on a 2-issue-per-cycle
// machine (one load-issue slot, one compute/consume slot, usable together
// in the same cycle): iteration i issues a load whose result becomes ready
// exactly `latency` cycles later, then a 1-cycle compute consumes it.
// Every iteration needs exactly 2 useful issue-events (1 load-issue + 1
// consume); with a 2-wide machine that is a hard floor of `n` cycles no
// matter how the loop is scheduled. "Stalls" are the cycles beyond that
// floor.
//
//   naive:      one iteration fully completes (issue -> wait `latency` ->
//               consume) before the next iteration's load is even issued.
//               Every iteration pays the full `latency` as stall time.
//   pipelined:  the next iteration's load is issued while earlier loads
//               are still in flight, so only the FIRST iteration ever
//               waits on an empty pipeline (the "fill") -- every later
//               result arrives just as the machine is ready to consume
//               it. The `latency`-cycle fill cost is paid exactly ONCE,
//               not once per iteration.
//
// Return {naive_stalls, pipelined_stalls} for the given n and latency.
// ============================================================================
StallCounts modeled_stall_counts(long long n, long long latency);
