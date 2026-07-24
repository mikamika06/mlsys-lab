#include "sol.hpp"
#include <sstream>

namespace {
void parse_op(const std::string& s, std::string& op, double& arg) {
    std::istringstream iss(s);
    iss >> op;
    if (op == "add" || op == "sub") iss >> arg;
}
}  // namespace

NaiveExpected compute_expected(const std::vector<std::string>& ops) {
    double state = 0.0;
    for (const auto& raw : ops) {
        std::string op;
        double arg = 0.0;
        parse_op(raw, op, arg);
        if (op == "add") {
            state += arg;
        } else if (op == "sub") {
            state -= arg;
        } else if (op == "div0") {
            return NaiveExpected{false, 0.0, 1};
        } else if (op == "overflow") {
            return NaiveExpected{false, 0.0, 2};
        }
    }
    return NaiveExpected{true, state, 0};
}

double compute_exceptions(const std::vector<std::string>& ops) {
    double state = 0.0;
    for (const auto& raw : ops) {
        std::string op;
        double arg = 0.0;
        parse_op(raw, op, arg);
        if (op == "add") {
            state += arg;
        } else if (op == "sub") {
            state -= arg;
        } else if (op == "div0") {
            throw OpFailure(1);
        } else if (op == "overflow") {
            throw OpFailure(2);
        }
    }
    return state;
}

long naive_expected_size() { return (long)sizeof(NaiveExpected); }
