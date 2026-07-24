#include "sol.hpp"

int grow_capacity(int capacity, double growth_factor) {
    int grown = static_cast<int>(capacity * growth_factor);
    return grown > 1 ? grown : 1;
}
