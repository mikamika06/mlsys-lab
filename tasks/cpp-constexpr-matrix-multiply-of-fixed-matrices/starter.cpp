#include "sol.hpp"

// TODO: implement both multiplies as real constexpr functions (must be
// usable in a constant expression -- see the self-check below). C = A * B,
// row-major.
constexpr Mat<2, 2> const_matmul_2x3(const Mat<2, 3>& A, const Mat<3, 2>& B) {
    (void)A;
    (void)B;
    // your code here
    return Mat<2, 2>{};
}

constexpr Mat<4, 4> const_matmul_4x4(const Mat<4, 4>& A, const Mat<4, 4>& B) {
    (void)A;
    (void)B;
    // your code here
    return Mat<4, 4>{};
}

// A `constexpr` function is implicitly `inline`; if the only use of it in
// this translation unit is fully constant-folded away (as in the
// static_asserts below), clang may not emit a linkable definition for it at
// all, leaving main.cpp's ordinary runtime call unresolved at link time.
// Taking its address (and marking that reachable even under optimization)
// forces a real out-of-line definition to exist. Do not edit below this
// line.
using Fn2x3 = Mat<2, 2> (*)(const Mat<2, 3>&, const Mat<3, 2>&);
using Fn4x4 = Mat<4, 4> (*)(const Mat<4, 4>&, const Mat<4, 4>&);
[[gnu::used]] static Fn2x3 g_keepalive_2x3 = &const_matmul_2x3;
[[gnu::used]] static Fn4x4 g_keepalive_4x4 = &const_matmul_4x4;

// Compile-time self-check: a constexpr function's definition must be
// visible in the translation unit that evaluates it in a constant
// expression, so this check has to live here (next to the definitions
// above) rather than in main.cpp, which only sees the declarations.
namespace {
constexpr Mat<2, 3> kA1 = {{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
constexpr Mat<3, 2> kB1 = {{1.0, 0.0, 0.0, 1.0, 1.0, 1.0}};
constexpr Mat<2, 2> kC1 = const_matmul_2x3(kA1, kB1);
static_assert(kC1.d[0] == 4.0 && kC1.d[1] == 5.0 && kC1.d[2] == 10.0 && kC1.d[3] == 11.0,
              "const_matmul_2x3 must be a constant expression computing A*B");

constexpr Mat<4, 4> kA2 = {{0.0, 2.0, -1.0, -1.0, 3.0, 3.0, 3.0, 4.0,
                            4.0, 3.0, 2.0, 0.0, -4.0, 3.0, -4.0, 3.0}};
constexpr Mat<4, 4> kB2 = {{2.0, -1.0, 1.0, 4.0, 4.0, 3.0, 1.0, -4.0,
                            -4.0, -2.0, 4.0, 2.0, 0.0, 3.0, -1.0, -1.0}};
constexpr Mat<4, 4> kC2 = const_matmul_4x4(kA2, kB2);
static_assert(kC2.d[0] == 12.0 && kC2.d[1] == 5.0 && kC2.d[2] == -1.0 && kC2.d[3] == -9.0 &&
                  kC2.d[4] == 6.0 && kC2.d[5] == 12.0 && kC2.d[6] == 14.0 && kC2.d[7] == 2.0 &&
                  kC2.d[8] == 12.0 && kC2.d[9] == 1.0 && kC2.d[10] == 15.0 && kC2.d[11] == 8.0 &&
                  kC2.d[12] == 20.0 && kC2.d[13] == 30.0 && kC2.d[14] == -20.0 && kC2.d[15] == -39.0,
              "const_matmul_4x4 must be a constant expression computing A*B");
}  // namespace
