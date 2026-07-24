#include "sol.hpp"

double modeled_vector_speedup(int n, int width) {
    int vector_instrs = n / width + n % width;
    return static_cast<double>(n) / static_cast<double>(vector_instrs);
}
