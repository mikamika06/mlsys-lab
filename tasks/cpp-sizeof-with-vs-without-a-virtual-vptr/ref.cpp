#include "sol.hpp"

long virtual_sizeof(long plain_size, long plain_align) {
    long align = plain_align > 8 ? plain_align : 8;
    long size = plain_size + 8;
    long rem = size % align;
    if (rem != 0) size += align - rem;
    return size;
}
