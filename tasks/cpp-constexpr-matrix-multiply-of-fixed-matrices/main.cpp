// FIXED driver. A `constexpr` function's *definition* must be visible in
// whichever translation unit actually evaluates it as a constant expression
// -- so the compile-time self-check lives inside ref.cpp/solve.cpp
// themselves (right next to the definitions they check; see the comment
// there). main.cpp just calls the finished functions at runtime, on its own
// matrices, and prints the product so the grader has numbers to diff.
#include <cstdio>

#include "sol.hpp"

int main() {
    Mat<2, 3> A1 = {{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
    Mat<3, 2> B1 = {{1.0, 0.0, 0.0, 1.0, 1.0, 1.0}};
    Mat<2, 2> C1 = const_matmul_2x3(A1, B1);
    for (int i = 0; i < 4; i++) printf("%.6f\n", C1.d[i]);

    Mat<4, 4> A2 = {{0.0, 2.0, -1.0, -1.0, 3.0, 3.0, 3.0, 4.0,
                     4.0, 3.0, 2.0, 0.0, -4.0, 3.0, -4.0, 3.0}};
    Mat<4, 4> B2 = {{2.0, -1.0, 1.0, 4.0, 4.0, 3.0, 1.0, -4.0,
                     -4.0, -2.0, 4.0, 2.0, 0.0, 3.0, -1.0, -1.0}};
    Mat<4, 4> C2 = const_matmul_4x4(A2, B2);
    for (int i = 0; i < 16; i++) printf("%.6f\n", C2.d[i]);

    return 0;
}
