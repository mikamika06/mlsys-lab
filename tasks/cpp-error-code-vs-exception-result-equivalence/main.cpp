#include <cstdio>
#include <string>
#include <vector>
#include "sol.hpp"

// FIXED driver. Do not edit. Runs five fixed op sequences through both
// models and prints the results, including whether compute_exceptions
// actually threw (caught here as OpFailure).
int main() {
    std::vector<std::vector<std::string>> cases = {
        {},
        {"add 1.5", "add 2.5"},
        {"add 1.0", "div0", "add 3.0"},
        {"sub 2.0", "overflow"},
        {"div0", "overflow"},
    };

    printf("naive_expected_size=%ld\n", naive_expected_size());

    for (const auto& ops : cases) {
        NaiveExpected e = compute_expected(ops);
        if (e.has_value) {
            printf("expected: has_value=1 val=%.6f\n", e.val);
        } else {
            printf("expected: has_value=0 err=%d\n", e.err);
        }

        try {
            double v = compute_exceptions(ops);
            printf("exceptions: ok val=%.6f\n", v);
        } catch (const OpFailure& f) {
            printf("exceptions: threw err=%d\n", f.err);
        }
    }
    return 0;
}
