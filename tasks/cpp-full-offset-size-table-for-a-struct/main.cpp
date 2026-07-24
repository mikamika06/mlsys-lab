// Fixed driver: ten real structs, one per test case, whose true layout is
// read straight off the real compiler via offsetof/sizeof (not a
// simulator). The candidate's struct_layout() prediction is printed
// directly, so any divergence from the true layout shows up as a byte
// difference in stdout.
#include "sol.hpp"
#include <cstddef>
#include <cstdio>

struct S1 { char f0; int f1; double f2; };
struct S2 { char f0; char f1; };
struct S3 { double f0; char f1; int f2; };
struct S4 { short f0; double f1; };
struct S5 { char f0; };
struct S6 { int f0; int f1; char f2; };
struct S7 { char f0; short f1; char f2; int f3; };
struct S8 { void* f0; char f1; void* f2; };
struct S9 { float f0; double f1; bool f2; };
struct S10 { long long f0; char f1; short f2; };

static void run_case(const FieldType* fields, int n, const int* real_off,
                      const int* real_size, int real_total) {
    FieldLayout out[8];
    int total = -1;
    struct_layout(fields, n, out, &total);
    int match = 1;
    for (int i = 0; i < n; i++) {
        printf("%d %d ", out[i].offset, out[i].size);
        if (out[i].offset != real_off[i] || out[i].size != real_size[i]) match = 0;
    }
    if (total != real_total) match = 0;
    printf("| %d | match=%d\n", total, match);
}

int main() {
    {
        FieldType f[] = {FieldType::Char, FieldType::Int, FieldType::Double};
        int off[] = {(int)offsetof(S1, f0), (int)offsetof(S1, f1), (int)offsetof(S1, f2)};
        int sz[] = {(int)sizeof(char), (int)sizeof(int), (int)sizeof(double)};
        run_case(f, 3, off, sz, (int)sizeof(S1));
    }
    {
        FieldType f[] = {FieldType::Char, FieldType::Char};
        int off[] = {(int)offsetof(S2, f0), (int)offsetof(S2, f1)};
        int sz[] = {(int)sizeof(char), (int)sizeof(char)};
        run_case(f, 2, off, sz, (int)sizeof(S2));
    }
    {
        FieldType f[] = {FieldType::Double, FieldType::Char, FieldType::Int};
        int off[] = {(int)offsetof(S3, f0), (int)offsetof(S3, f1), (int)offsetof(S3, f2)};
        int sz[] = {(int)sizeof(double), (int)sizeof(char), (int)sizeof(int)};
        run_case(f, 3, off, sz, (int)sizeof(S3));
    }
    {
        FieldType f[] = {FieldType::Short, FieldType::Double};
        int off[] = {(int)offsetof(S4, f0), (int)offsetof(S4, f1)};
        int sz[] = {(int)sizeof(short), (int)sizeof(double)};
        run_case(f, 2, off, sz, (int)sizeof(S4));
    }
    {
        FieldType f[] = {FieldType::Char};
        int off[] = {(int)offsetof(S5, f0)};
        int sz[] = {(int)sizeof(char)};
        run_case(f, 1, off, sz, (int)sizeof(S5));
    }
    {
        FieldType f[] = {FieldType::Int, FieldType::Int, FieldType::Char};
        int off[] = {(int)offsetof(S6, f0), (int)offsetof(S6, f1), (int)offsetof(S6, f2)};
        int sz[] = {(int)sizeof(int), (int)sizeof(int), (int)sizeof(char)};
        run_case(f, 3, off, sz, (int)sizeof(S6));
    }
    {
        FieldType f[] = {FieldType::Char, FieldType::Short, FieldType::Char, FieldType::Int};
        int off[] = {(int)offsetof(S7, f0), (int)offsetof(S7, f1), (int)offsetof(S7, f2),
                     (int)offsetof(S7, f3)};
        int sz[] = {(int)sizeof(char), (int)sizeof(short), (int)sizeof(char), (int)sizeof(int)};
        run_case(f, 4, off, sz, (int)sizeof(S7));
    }
    {
        FieldType f[] = {FieldType::Pointer, FieldType::Char, FieldType::Pointer};
        int off[] = {(int)offsetof(S8, f0), (int)offsetof(S8, f1), (int)offsetof(S8, f2)};
        int sz[] = {(int)sizeof(void*), (int)sizeof(char), (int)sizeof(void*)};
        run_case(f, 3, off, sz, (int)sizeof(S8));
    }
    {
        FieldType f[] = {FieldType::Float, FieldType::Double, FieldType::Bool};
        int off[] = {(int)offsetof(S9, f0), (int)offsetof(S9, f1), (int)offsetof(S9, f2)};
        int sz[] = {(int)sizeof(float), (int)sizeof(double), (int)sizeof(bool)};
        run_case(f, 3, off, sz, (int)sizeof(S9));
    }
    {
        FieldType f[] = {FieldType::LongLong, FieldType::Char, FieldType::Short};
        int off[] = {(int)offsetof(S10, f0), (int)offsetof(S10, f1), (int)offsetof(S10, f2)};
        int sz[] = {(int)sizeof(long long), (int)sizeof(char), (int)sizeof(short)};
        run_case(f, 3, off, sz, (int)sizeof(S10));
    }
    return 0;
}
