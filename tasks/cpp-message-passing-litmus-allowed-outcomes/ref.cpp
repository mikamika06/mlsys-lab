#include "sol.hpp"
#include <algorithm>

std::vector<std::pair<int, int>> get_allowed_litmus_outcomes(
    std::memory_order write_mo, std::memory_order read_mo, int data_val) {
    bool sync_write = write_mo == std::memory_order_release ||
                       write_mo == std::memory_order_acq_rel ||
                       write_mo == std::memory_order_seq_cst;
    bool sync_read = read_mo == std::memory_order_acquire ||
                      read_mo == std::memory_order_acq_rel ||
                      read_mo == std::memory_order_seq_cst;
    bool synchronized = sync_write && sync_read;

    std::vector<std::pair<int, int>> outcomes = {
        {0, 0}, {0, data_val}, {1, data_val},
    };
    if (!synchronized) outcomes.push_back({1, 0});

    std::sort(outcomes.begin(), outcomes.end());
    outcomes.erase(std::unique(outcomes.begin(), outcomes.end()), outcomes.end());
    return outcomes;
}
