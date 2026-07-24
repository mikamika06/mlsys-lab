#include <cstdio>
#include <cstddef>
#include "sol.hpp"

// FIXED driver. For each case, asks the learner's compute_layout() for the
// size/offsets of a field-type sequence, and independently asks the REAL
// compiler for sizeof/offsetof of an equivalently-declared real struct —
// the actual ground truth, not a modeled one. Prints both side by side.

using FT = FieldType;

struct S1  { int a; int b; char c; };
struct S2  { void* a; char b; void* c; bool d; };
struct S3  { char a; int b; double c; };
struct S4  { char a; char b; };
struct S5  { double a; char b; int c; };
struct S6  { short a; double b; };
struct S7  { char a; };
struct S8  { void* a; };
struct S9  { void* a; bool b; int c; double d; };
struct S10 { long a; char b; long c; };
struct S11 { float a; float b; char c; };
struct S12 { short a; short b; short c; short d; };
struct S13 { bool a; };
struct S14 { long long a; void* b; char c; };

static void report(int n, int computed_size, const int* computed_off,
                    int real_size, const int* real_off) {
    printf("%d ", computed_size);
    for (int i = 0; i < n; ++i) printf("%d ", computed_off[i]);
    printf("| %d ", real_size);
    for (int i = 0; i < n; ++i) printf("%d ", real_off[i]);
    printf("\n");
}

int main() {
    {
        FT f[] = {FT::Int, FT::Int, FT::Char};
        int off[3];
        int sz = compute_layout(f, 3, off);
        int real_off[3] = {(int)offsetof(S1, a), (int)offsetof(S1, b), (int)offsetof(S1, c)};
        report(3, sz, off, (int)sizeof(S1), real_off);
    }
    {
        FT f[] = {FT::Pointer, FT::Char, FT::Pointer, FT::Bool};
        int off[4];
        int sz = compute_layout(f, 4, off);
        int real_off[4] = {(int)offsetof(S2, a), (int)offsetof(S2, b), (int)offsetof(S2, c), (int)offsetof(S2, d)};
        report(4, sz, off, (int)sizeof(S2), real_off);
    }
    {
        FT f[] = {FT::Char, FT::Int, FT::Double};
        int off[3];
        int sz = compute_layout(f, 3, off);
        int real_off[3] = {(int)offsetof(S3, a), (int)offsetof(S3, b), (int)offsetof(S3, c)};
        report(3, sz, off, (int)sizeof(S3), real_off);
    }
    {
        FT f[] = {FT::Char, FT::Char};
        int off[2];
        int sz = compute_layout(f, 2, off);
        int real_off[2] = {(int)offsetof(S4, a), (int)offsetof(S4, b)};
        report(2, sz, off, (int)sizeof(S4), real_off);
    }
    {
        FT f[] = {FT::Double, FT::Char, FT::Int};
        int off[3];
        int sz = compute_layout(f, 3, off);
        int real_off[3] = {(int)offsetof(S5, a), (int)offsetof(S5, b), (int)offsetof(S5, c)};
        report(3, sz, off, (int)sizeof(S5), real_off);
    }
    {
        FT f[] = {FT::Short, FT::Double};
        int off[2];
        int sz = compute_layout(f, 2, off);
        int real_off[2] = {(int)offsetof(S6, a), (int)offsetof(S6, b)};
        report(2, sz, off, (int)sizeof(S6), real_off);
    }
    {
        FT f[] = {FT::Char};
        int off[1];
        int sz = compute_layout(f, 1, off);
        int real_off[1] = {(int)offsetof(S7, a)};
        report(1, sz, off, (int)sizeof(S7), real_off);
    }
    {
        FT f[] = {FT::Pointer};
        int off[1];
        int sz = compute_layout(f, 1, off);
        int real_off[1] = {(int)offsetof(S8, a)};
        report(1, sz, off, (int)sizeof(S8), real_off);
    }
    {
        FT f[] = {FT::Pointer, FT::Bool, FT::Int, FT::Double};
        int off[4];
        int sz = compute_layout(f, 4, off);
        int real_off[4] = {(int)offsetof(S9, a), (int)offsetof(S9, b), (int)offsetof(S9, c), (int)offsetof(S9, d)};
        report(4, sz, off, (int)sizeof(S9), real_off);
    }
    {
        FT f[] = {FT::Long, FT::Char, FT::Long};
        int off[3];
        int sz = compute_layout(f, 3, off);
        int real_off[3] = {(int)offsetof(S10, a), (int)offsetof(S10, b), (int)offsetof(S10, c)};
        report(3, sz, off, (int)sizeof(S10), real_off);
    }
    {
        FT f[] = {FT::Float, FT::Float, FT::Char};
        int off[3];
        int sz = compute_layout(f, 3, off);
        int real_off[3] = {(int)offsetof(S11, a), (int)offsetof(S11, b), (int)offsetof(S11, c)};
        report(3, sz, off, (int)sizeof(S11), real_off);
    }
    {
        FT f[] = {FT::Short, FT::Short, FT::Short, FT::Short};
        int off[4];
        int sz = compute_layout(f, 4, off);
        int real_off[4] = {(int)offsetof(S12, a), (int)offsetof(S12, b), (int)offsetof(S12, c), (int)offsetof(S12, d)};
        report(4, sz, off, (int)sizeof(S12), real_off);
    }
    {
        FT f[] = {FT::Bool};
        int off[1];
        int sz = compute_layout(f, 1, off);
        int real_off[1] = {(int)offsetof(S13, a)};
        report(1, sz, off, (int)sizeof(S13), real_off);
    }
    {
        FT f[] = {FT::LongLong, FT::Pointer, FT::Char};
        int off[3];
        int sz = compute_layout(f, 3, off);
        int real_off[3] = {(int)offsetof(S14, a), (int)offsetof(S14, b), (int)offsetof(S14, c)};
        report(3, sz, off, (int)sizeof(S14), real_off);
    }
    return 0;
}
