#pragma once

// Run 4 branch predictors over a fixed trace `outcomes[0..n)` (each entry
// 0 = not-taken, 1 = taken) and write each predictor's total mispredict
// count into out[0..3]:
//
//   out[0]  Always-taken: predicts "taken" every time. Mispredicts on
//           every 0 in the trace.
//
//   out[1]  1-bit last-outcome: predicts whatever the PREVIOUS actual
//           outcome was (predict starts at 0, i.e. not-taken, before any
//           branch has been seen). After each branch, the predictor for
//           next time becomes this branch's actual outcome.
//
//   out[2]  Single 2-bit saturating counter, shared by the whole trace.
//           Counter starts at 1 ("weakly not-taken"). Predict "taken" iff
//           counter >= 2. After seeing the actual outcome: if taken,
//           counter = min(3, counter + 1); if not-taken,
//           counter = max(0, counter - 1).
//
//   out[3]  gshare: a table of 2^hist_bits independent 2-bit saturating
//           counters (same update rule as out[2]; every counter starts at
//           1), indexed by a hist_bits-wide global history register
//           (starts at 0, holding the hist_bits most recent actual
//           outcomes, most recent in the low bit). For each branch:
//           index = history & ((1 << hist_bits) - 1); predict from
//           table[index]; update table[index] with the actual outcome;
//           then history = ((history << 1) | actual) & ((1 << hist_bits) - 1).
//
// A predictor mispredicts on step i whenever its prediction != outcomes[i].
void predictor_mispredicts(const int* outcomes, int n, int hist_bits, int* out);
