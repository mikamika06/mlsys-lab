#pragma once
// Count branch mispredictions using an independent 2-bit saturating
// counter per branch id.
//
// Per-branch state machine:
//   0 = strongly not-taken   1 = weakly not-taken
//   2 = weakly taken         3 = strongly taken
// Every branch's counter starts at state 1 (weakly not-taken). The
// predictor predicts "taken" iff its branch's current state >= 2. After
// seeing the actual outcome, the counter saturates toward the outcome:
//   state = min(3, state + 1)   if the branch was actually taken
//   state = max(0, state - 1)   if the branch was actually not taken
//
// branch_ids[i]/outcomes[i] (for 0 <= i < n) is the i-th branch event in
// program order: it belongs to branch branch_ids[i]
// (0 <= branch_ids[i] < num_branches), and outcomes[i] is 1 if that
// branch was actually taken this time, 0 otherwise. Predict BEFORE
// updating, using each branch's state as of just before this event.
//
// Return the total number of mispredictions across all n events.
int count_mispredicts(const int* branch_ids, const int* outcomes, int n, int num_branches);
