#include "sol.hpp"

// Reference: independent 2-bit saturating counter per branch id.
int count_mispredicts(const int* branch_ids, const int* outcomes, int n, int num_branches) {
    int* state = new int[num_branches];
    for (int b = 0; b < num_branches; b++) state[b] = 1;  // weakly not-taken

    int mispredicts = 0;
    for (int i = 0; i < n; i++) {
        int b = branch_ids[i];
        int predicted_taken = state[b] >= 2 ? 1 : 0;
        int actual_taken = outcomes[i];
        if (predicted_taken != actual_taken) mispredicts++;

        if (actual_taken) {
            state[b] = state[b] + 1 > 3 ? 3 : state[b] + 1;
        } else {
            state[b] = state[b] - 1 < 0 ? 0 : state[b] - 1;
        }
    }

    delete[] state;
    return mispredicts;
}
