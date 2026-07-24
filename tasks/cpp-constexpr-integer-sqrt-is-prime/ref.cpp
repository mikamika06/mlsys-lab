#include "sol.hpp"

int integer_sqrt(int n) {
    if (n < 2) return n;
    int lo = 1, hi = n, answer = 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (mid <= n / mid) {
            answer = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return answer;
}

bool is_prime(int n) {
    if (n < 2) return false;
    int limit = integer_sqrt(n);
    for (int d = 2; d <= limit; d++) {
        if (n % d == 0) return false;
    }
    return true;
}
