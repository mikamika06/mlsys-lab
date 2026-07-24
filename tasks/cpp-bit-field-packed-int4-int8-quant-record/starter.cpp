#include "sol.hpp"

// TODO: pack weights[0..32) into 16 nibble-pairs and write scale,
// zero_point, and the packed weights into `out` at QuantBlock's real
// (compiler-decided) field offsets. See sol.hpp for the exact contract.
void pack_quant_block(int8_t scale, int32_t zero_point, const int weights[32],
                       uint8_t* out, int out_len) {
    (void)scale;
    (void)zero_point;
    (void)weights;
    (void)out;
    (void)out_len;
}
