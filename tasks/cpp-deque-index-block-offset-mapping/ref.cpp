#include "sol.hpp"

std::vector<std::pair<long, long>> deque_mapping(long elem_size, long first_offset,
                                                   const std::vector<long>& indices) {
    long N = 512 / elem_size;
    if (N < 1) N = 1;

    std::vector<std::pair<long, long>> res;
    res.reserve(indices.size());
    for (long i : indices) {
        long absolute = first_offset + i;
        res.push_back({absolute / N, absolute % N});
    }
    return res;
}
