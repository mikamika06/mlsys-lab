#include "sol.hpp"

Tracked reduce_balanced_tree(const Tracked* x, int n) {
    if (n == 1) return x[0];
    int mid = n / 2;
    Tracked left = reduce_balanced_tree(x, mid);
    Tracked right = reduce_balanced_tree(x + mid, n - mid);
    return left + right;
}
