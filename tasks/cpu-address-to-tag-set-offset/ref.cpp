#include "sol.hpp"

AddrDecomp decompose_address(unsigned long addr, int line_bytes, int sets, int ways) {
    (void)ways;
    unsigned long line_index = addr / static_cast<unsigned long>(line_bytes);
    AddrDecomp d;
    d.offset = addr % static_cast<unsigned long>(line_bytes);
    d.set_index = line_index % static_cast<unsigned long>(sets);
    d.tag = line_index / static_cast<unsigned long>(sets);
    return d;
}
