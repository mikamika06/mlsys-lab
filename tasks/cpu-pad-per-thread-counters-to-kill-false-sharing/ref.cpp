#include "sol.hpp"

int counter_pad_bytes() {
    return 56;  // 8-byte counter + 56 bytes padding = 64-byte stride
}
