#pragma once
// A fixed-size row-major matrix, used purely as a COMPILE-TIME value: every
// object of this type that main.cpp builds is `constexpr`, and the whole
// point of this task is that the multiply below must be able to run during
// translation, not just at runtime.
template <int R, int C>
struct Mat {
    double d[R * C];
};

// Multiply A (2x3) by B (3x2) and return the 2x2 product C = A * B,
// row-major: C.d[i*2+j] = sum_k A.d[i*3+k] * B.d[k*2+j].
//
// Must be declared `constexpr` and must actually be usable in a constant
// expression -- main.cpp forces this by assigning the result to a
// `constexpr` variable and `static_assert`-ing every entry against the true
// product. If your implementation isn't a valid constant expression (e.g.
// it allocates, throws on this path, or otherwise can't be evaluated at
// compile time), the build fails to compile, which fails the gate exactly
// like a wrong numeric answer would.
constexpr Mat<2, 2> const_matmul_2x3(const Mat<2, 3>& A, const Mat<3, 2>& B);

// Multiply A (4x4) by B (4x4) and return the 4x4 product C = A * B,
// row-major. Same constexpr requirement as above.
constexpr Mat<4, 4> const_matmul_4x4(const Mat<4, 4>& A, const Mat<4, 4>& B);
