#include "sol.hpp"

// TODO: reassociate into a balanced binary tree: split [0,n) into two
// halves, call reduce_balanced_tree recursively on each half, then
// combine the two results with one operator+. Do not just sum
// everything into one running Tracked value serially -- that has the
// correct .value but a much longer .depth (n-1 instead of ceil(log2(n))).
Tracked reduce_balanced_tree(const Tracked* x, int n) {
    Tracked total{0.0, 0};
    for (int i = 0; i < n; i++) {
        total = total + x[i];
    }
    return total;
}
