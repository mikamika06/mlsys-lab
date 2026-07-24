#include "sol.hpp"

// TODO: for i in [0, record_count), touch(aos_base + i*field_count*4 +
// field_index*4) then touch(soa_out_base + i*4) -- exactly once each. See
// sol.hpp.
void aos_field_to_soa(long aos_base, long soa_out_base, int record_count,
                       int field_count, int field_index) {
    (void)aos_base; (void)soa_out_base; (void)record_count;
    (void)field_count; (void)field_index;
    // your code here
}
