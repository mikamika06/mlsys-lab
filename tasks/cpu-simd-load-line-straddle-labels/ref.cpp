#include "sol.hpp"

bool straddles_line(uint64_t base_addr, int width_bytes, int line_bytes) {
    uint64_t offset_in_line = base_addr % static_cast<uint64_t>(line_bytes);
    return offset_in_line + static_cast<uint64_t>(width_bytes) > static_cast<uint64_t>(line_bytes);
}
