#include <cstdio>
#include "sol.hpp"

int main() {
    int sc = allowed_outcomes(false);   // seq_cst  -> (0,0) is forbidden
    int sb = allowed_outcomes(true);    // relaxed  -> (0,0) becomes observable

    int counts[4] = {0, 0, 0, 0};
    sc_outcome_histogram(counts);

    printf("%d\n", sc);
    printf("%d\n", sb);
    printf("%d %d %d %d\n", counts[0], counts[1], counts[2], counts[3]);
    return 0;
}
