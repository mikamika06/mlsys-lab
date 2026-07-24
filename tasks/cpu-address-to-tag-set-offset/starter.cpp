#include "sol.hpp"

AddrDecomp decompose_address(unsigned long addr, int line_bytes, int sets, int ways) {
    // your code here
    AddrDecomp d;
    d.tag = 0;
    d.set_index = 0;
    d.offset = 0;
    return d;
}
