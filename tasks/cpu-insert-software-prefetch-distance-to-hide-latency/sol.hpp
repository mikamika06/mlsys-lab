#pragma once

// LEARNER IMPLEMENTS.
//
// Model a streaming loop over `n` array elements large enough that every
// element is a cold DRAM access (no cache reuse in play here). Cycle
// budget: each loop iteration takes `cycles_per_iter` cycles of work,
// and a demand memory access takes `latency_cycles` cycles to complete
// once issued.
//
// A SOFTWARE PREFETCH inserted `distance` iterations ahead means: while
// executing iteration i, the loop also issues a prefetch for the data
// iteration (i + distance) will need. That prefetch then has
// `distance * cycles_per_iter` cycles of loop work to complete in
// before iteration (i + distance) actually demands the data.
//
// Return the number of iterations (out of n) whose demand access
// STALLS -- i.e. is NOT fully hidden by a prefetch:
//   - the first min(distance, n) iterations always stall: there is no
//     earlier iteration to have issued their prefetch from yet (the
//     pipeline hasn't "warmed up").
//   - every iteration at or past `distance` stalls too, UNLESS
//     distance * cycles_per_iter >= latency_cycles (the prefetch gets
//     enough of a head start to finish before it's needed) -- in that
//     case none of them stall.
int count_stalls(int n, int distance, int latency_cycles, int cycles_per_iter);
