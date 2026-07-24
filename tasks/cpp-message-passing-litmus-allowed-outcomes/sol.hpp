#pragma once
#include <atomic>
#include <vector>
#include <utility>

// ---------------------------------------------------------------------------
// LEARNER IMPLEMENTS.
//
// Enumerate the allowed final register outcomes (r1, r2) of the classic
// Message-Passing litmus test on two initially-zero shared variables:
//
//   Producer T1:  data_field = data_val;
//                 flag_field.store(1, write_mo);
//   Consumer T2:  r1 = flag_field.load(read_mo);
//                 r2 = data_field;
//
// If `write_mo` is one of {release, acq_rel, seq_cst} AND `read_mo` is one
// of {acquire, acq_rel, seq_cst}, the store synchronizes-with the load:
// once T2 observes r1 == 1, everything T1 did before the store (including
// writing data_field) is guaranteed visible -- so the outcome (1, 0) is
// impossible. In every other combination of memory orders, no
// happens-before edge is established between the two threads and all
// four combinations of r1 in {0, 1}, r2 in {0, data_val} are possible
// (the store and the write to data_field can appear reordered from T2's
// point of view).
//
// Return the SORTED (lexicographic on (r1, r2)) list of unique allowed
// outcomes.
// ---------------------------------------------------------------------------
std::vector<std::pair<int, int>> get_allowed_litmus_outcomes(
    std::memory_order write_mo, std::memory_order read_mo, int data_val);
