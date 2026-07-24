#include "sol.hpp"

void aos_field_to_soa(long aos_base, long soa_out_base, int record_count,
                       int field_count, int field_index) {
    for (int i = 0; i < record_count; i++) {
        long src = aos_base + (long)i * field_count * 4 + (long)field_index * 4;
        long dst = soa_out_base + (long)i * 4;
        touch(src);
        touch(dst);
    }
}
