#pragma once

// ============================================================================
// LEARNER implements both of these in solve.cpp.
//
// Unrolling a loop of N iterations by factor U replaces it with ceil(N/U)
// "outer" iterations, each doing U original iterations' worth of work.
// Two effects trade off against each other as U grows:
//
//   - Loop overhead (branch + counter update, modeled as a fixed cost
//     C_loop per OUTER iteration) is paid ceil(N/U) times -- larger U
//     means fewer outer iterations, so less total overhead.
//   - If U exceeds the R accumulator registers actually available, the
//     extra (U - R) accumulators spill to the stack, adding a modeled
//     C_spill penalty PER SPILLED REGISTER, paid on every one of the
//     ceil(N/U) outer iterations.
//
// unroll_cost(N, U, C_loop, R, C_spill):
//   Return ceil(N/U) * (C_loop + max(0, U - R) * C_spill).
//
// choose_best_unroll(N, max_U, C_loop, R, C_spill):
//   Try every U in [1, max_U] (inclusive), calling unroll_cost for each,
//   and return whichever U gives the SMALLEST cost (the SMALLEST such U
//   on a tie).
// ============================================================================
long unroll_cost(int N, int U, int C_loop, int R, int C_spill);
int choose_best_unroll(int N, int max_U, int C_loop, int R, int C_spill);
