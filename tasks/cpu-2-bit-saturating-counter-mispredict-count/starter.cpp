#include "sol.hpp"

// TODO: maintain one 2-bit saturating counter per branch id (all starting
// at state 1, weakly not-taken), predict "taken" iff state >= 2, count a
// mispredict whenever the prediction differs from the actual outcome, then
// update the counter toward the actual outcome.
int count_mispredicts(const int* branch_ids, const int* outcomes, int n, int num_branches) {
    (void)branch_ids;
    (void)outcomes;
    (void)n;
    (void)num_branches;
    // your code here
    return 0;
}
