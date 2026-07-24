#include <cstdio>
#include <cstring>
#include "sol.hpp"

static BufferObj make_buf(const char* dtype, int ndim, const int* shape, double fill) {
    BufferObj b;
    b.is_valid_buffer = true;
    b.dtype = dtype;
    b.ndim = ndim;
    int size = 1;
    for (int i = 0; i < ndim; i++) { b.shape[i] = shape[i]; size *= shape[i]; }
    b.size = size;
    for (int i = 0; i < size; i++) b.data[i] = fill;
    return b;
}

int main() {
    // ---- Build 10 fixture buffers ----
    int sh1[2] = {3, 3};
    BufferObj arr1 = make_buf("float32", 2, sh1, 1.0);                 // valid, sum=9

    int sh2[1] = {3};
    BufferObj arr2 = make_buf("int64", 1, sh2, 0.0);
    arr2.data[0] = 10; arr2.data[1] = 20; arr2.data[2] = 30;           // valid, sum=60

    int sh3[3] = {2, 4, 5};
    BufferObj arr3 = make_buf("float64", 3, sh3, 1.0);                 // valid, sum=40

    int sh4[2] = {10, 5};
    BufferObj arr4 = make_buf("int32", 2, sh4, 1.0);                   // valid, sum=50

    int sh5[2] = {2, 2};
    BufferObj arr5 = make_buf("float32", 2, sh5, 1.0);
    arr5.is_valid_buffer = false;                                      // not a buffer at all

    BufferObj arr6 = make_buf("float64", 2, sh1, 1.0);                 // dtype mismatch vs expected float32

    BufferObj arr7 = make_buf("float32", 2, sh1, 1.0);                 // ndim mismatch: expect 3D

    int sh8[2] = {3, 4};
    BufferObj arr8 = make_buf("float32", 2, sh8, 1.0);                 // shape mismatch vs expected (3,3)

    int sh9[2] = {10, 4};
    BufferObj arr9 = make_buf("float32", 2, sh9, 1.0);                 // shape mismatch even with wildcard dim0

    int sh10[2] = {4, 4};
    BufferObj arr10 = make_buf("float64", 2, sh10, 2.5);               // valid wildcard dim1, sum=40

    struct Case {
        const BufferObj* arr;
        const char* exp_dtype;
        const int* exp_shape;
        int exp_ndim;
    };

    int esh1[2]  = {3, 3};
    int esh2[1]  = {-1};
    int esh3[3]  = {2, 4, 5};
    int esh4[2]  = {-1, 5};
    int esh5[2]  = {2, 2};
    int esh6[2]  = {3, 3};
    int esh7[3]  = {3, 3, 1};
    int esh8[2]  = {3, 3};
    int esh9[2]  = {-1, 5};
    int esh10[2] = {4, -1};

    Case cases[10] = {
        {&arr1,  "float32", esh1,  2},
        {&arr2,  "int64",   esh2,  1},
        {&arr3,  "float64", esh3,  3},
        {&arr4,  "int32",   esh4,  2},
        {&arr5,  "float32", esh5,  2},
        {&arr6,  "float32", esh6,  2},
        {&arr7,  "float32", esh7,  3},
        {&arr8,  "float32", esh8,  2},
        {&arr9,  "float32", esh9,  2},
        {&arr10, "float64", esh10, 2},
    };

    for (int i = 0; i < 10; i++) {
        try {
            double v = validate_buffer(*cases[i].arr, cases[i].exp_dtype, cases[i].exp_shape, cases[i].exp_ndim);
            printf("case%d: ok value=%.6f\n", i, v);
        } catch (const TypeErrorSim&) {
            printf("case%d: TypeError\n", i);
        } catch (const ValueErrorSim&) {
            printf("case%d: ValueError\n", i);
        } catch (...) {
            printf("case%d: UnknownError\n", i);
        }
    }
    return 0;
}
