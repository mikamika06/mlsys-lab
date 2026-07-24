#include "sol.hpp"

// TODO: implement all three. See sol.hpp for the exact op semantics and
// the required success/failure behavior of each model.
NaiveExpected compute_expected(const std::vector<std::string>& ops) {
    (void)ops;
    return NaiveExpected{false, 0.0, 0};
}

double compute_exceptions(const std::vector<std::string>& ops) {
    (void)ops;
    return 0.0;
}

long naive_expected_size() {
    return 0;
}
