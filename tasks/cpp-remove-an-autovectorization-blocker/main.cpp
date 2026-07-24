// Fixed driver: four fixed (struct_size, field_offset, n) cases, whose
// struct_size/field_offset were derived by hand from the same LP64
// natural-alignment rules a real compiler applies (char=1, short=2,
// int/float=4, long/double/pointer=8, alignment == size).
#include "sol.hpp"
#include <cstdio>
#include <cstring>
#include <vector>

static void run_case(int struct_size, int field_offset, int n) {
    std::vector<unsigned char> buf(static_cast<size_t>(n) * struct_size, 0);
    for (int i = 0; i < n; i++) {
        double v = static_cast<double>(i + 1);
        std::memcpy(buf.data() + static_cast<size_t>(i) * struct_size + field_offset, &v, sizeof(double));
    }
    std::vector<double> out(n);
    optimize_vector_loop(buf.data(), n, struct_size, field_offset, out.data());
    for (int i = 0; i < n; i++) {
        printf("%.6f ", out[i]);
    }
    printf("\n");
}

int main() {
    // fields=["int","double","float"], shape (4,4) -> n=16
    run_case(24, 8, 16);
    // fields=["char","double"], shape (2,8) -> n=16
    run_case(16, 8, 16);
    // fields=["short","short","double","int"], shape (3,5) -> n=15
    run_case(24, 2, 15);
    // fields=["pointer","double"], shape (6,2) -> n=12
    run_case(16, 8, 12);
    return 0;
}
