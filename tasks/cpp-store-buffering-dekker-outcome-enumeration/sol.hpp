#pragma once
// Store-buffering (Dekker) litmus test. Two shared atomics start at x = y = 0:
//
//   Thread A:  x.store(1);  r1 = y.load();
//   Thread B:  y.store(1);  r2 = x.load();
//
// Encode a final outcome (r1, r2) as the 2-bit index  (r1 << 1) | r2 :
//   idx 0 = (0,0)   idx 1 = (0,1)   idx 2 = (1,0)   idx 3 = (1,1)
//
// allowed_outcomes: return a 4-bit mask; bit i is set iff outcome i is observable
// under a conforming C++ implementation.
//   store_buffering == false : every operation uses memory_order_seq_cst, so you
//                              enumerate every interleaving of the four operations
//                              over a single sequentially-consistent memory.
//   store_buffering == true  : the stores are relaxed, so each thread may hold its
//                              store in a private buffer and read the other location
//                              before that store becomes globally visible (the
//                              classic store-buffering / x86-TSO relaxation).
int allowed_outcomes(bool store_buffering);

// sc_outcome_histogram: over ALL sequentially-consistent interleavings of the four
// operations (there are C(4,2) = 6 of them, respecting per-thread program order),
// count how many produce each outcome. Fill counts[i] with the number of
// interleavings whose result has index i (as defined above).
void sc_outcome_histogram(int counts[4]);
