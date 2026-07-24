#include "sol.hpp"

// TODO: enumerate the allowed (r1, r2) outcomes for this write_mo/read_mo
// pair. See sol.hpp for the exact synchronizes-with rule.
std::vector<std::pair<int, int>> get_allowed_litmus_outcomes(
    std::memory_order write_mo, std::memory_order read_mo, int data_val) {
    (void)write_mo;
    (void)read_mo;
    (void)data_val;
    return {};
}
