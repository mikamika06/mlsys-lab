#include "sol.hpp"

// TODO: classify the 10 fixed access patterns documented in sol.hpp using
// real offsetof() on ThreadState and the 64-byte cache-line rule.
std::pair<std::vector<bool>, long> classify_false_sharing() {
    return {std::vector<bool>(10, false), 0};
}
