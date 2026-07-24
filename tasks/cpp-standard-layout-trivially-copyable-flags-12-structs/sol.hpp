#pragma once
// Classify twelve C++ structs S1..S12 (their exact definitions are given in
// task.md) by two ABI properties each, and write 24 bits into out[0..23].
//
// For struct number k (1-based, k = 1..12):
//   out[2*(k-1)]     = 1 if Sk is standard-layout        (std::is_standard_layout),   else 0
//   out[2*(k-1) + 1] = 1 if Sk is trivially-copyable      (std::is_trivially_copyable), else 0
//
// So the layout is two bits per struct, standard-layout first, in order S1..S12:
//   out = [ SL(S1), TC(S1), SL(S2), TC(S2), ..., SL(S12), TC(S12) ]
void classify(int out[24]);
