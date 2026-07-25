#include "sol.hpp"

void soa_field_touch(int N, long soa_base) {
    for (int i = 0; i < N; i++) {
        touch(soa_base + (long)i * 4);
    }
}
