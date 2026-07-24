#include "sol.hpp"

void naive_matmul(int N, long a_base, long b_base, long c_base) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                touch_byte(a_base + (long)(i * N + k) * 4);
                touch_byte(b_base + (long)(k * N + j) * 4);
                touch_byte(c_base + (long)(i * N + j) * 4);
            }
        }
    }
}

// stride = the ORIGINAL matrix's row length in elements, fixed across
// the whole recursion; n = the current quadrant's side length.
static void recurse(int n, long a_base, long b_base, long c_base, int stride) {
    if (n <= 8) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    touch_byte(a_base + (long)(i * stride + k) * 4);
                    touch_byte(b_base + (long)(k * stride + j) * 4);
                    touch_byte(c_base + (long)(i * stride + j) * 4);
                }
            }
        }
        return;
    }

    int h = n / 2;
    long rowbytes = (long)h * stride * 4;
    long colbytes = (long)h * 4;

    long a11 = a_base,             a12 = a_base + colbytes;
    long a21 = a_base + rowbytes,  a22 = a21 + colbytes;
    long b11 = b_base,             b12 = b_base + colbytes;
    long b21 = b_base + rowbytes,  b22 = b21 + colbytes;
    long c11 = c_base,             c12 = c_base + colbytes;
    long c21 = c_base + rowbytes,  c22 = c21 + colbytes;

    // C11 += A11*B11 + A12*B21
    recurse(h, a11, b11, c11, stride);
    recurse(h, a12, b21, c11, stride);
    // C12 += A11*B12 + A12*B22
    recurse(h, a11, b12, c12, stride);
    recurse(h, a12, b22, c12, stride);
    // C21 += A21*B11 + A22*B21
    recurse(h, a21, b11, c21, stride);
    recurse(h, a22, b21, c21, stride);
    // C22 += A21*B12 + A22*B22
    recurse(h, a21, b12, c22, stride);
    recurse(h, a22, b22, c22, stride);
}

void recursive_matmul(int N, long a_base, long b_base, long c_base) {
    recurse(N, a_base, b_base, c_base, N);
}
