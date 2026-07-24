#include "sol.hpp"

// TODO: N = max(1, 512 / elem_size); for each i in indices, absolute =
// first_offset + i, append {absolute / N, absolute % N}. See sol.hpp.
std::vector<std::pair<long, long>> deque_mapping(long elem_size, long first_offset,
                                                   const std::vector<long>& indices) {
    (void)elem_size;
    (void)first_offset;
    (void)indices;
    return {};
}
