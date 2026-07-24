#include <cstdio>
#include "sol.hpp"

// Five real structs -- their field order IS the layout the grader checks
// against, via the compiler's own sizeof(), not a hand-derived number.
struct S1 { char a; int b; char c; double d; };
struct S2 { double a; char b; int c; short d; };
struct S3 { char a; char b; char c; char d; };
struct S4 { char a; long long b; char c; };
struct S5 { int a; char b; int c; char d; short e; };

int main() {
    // Field size/alignment arrays mirror each struct's members, in the
    // same declared order, via sizeof/alignof -- not hardcoded numbers.
    int s1_sizes[]  = {(int)sizeof(char), (int)sizeof(int), (int)sizeof(char), (int)sizeof(double)};
    int s1_aligns[] = {(int)alignof(char), (int)alignof(int), (int)alignof(char), (int)alignof(double)};

    int s2_sizes[]  = {(int)sizeof(double), (int)sizeof(char), (int)sizeof(int), (int)sizeof(short)};
    int s2_aligns[] = {(int)alignof(double), (int)alignof(char), (int)alignof(int), (int)alignof(short)};

    int s3_sizes[]  = {(int)sizeof(char), (int)sizeof(char), (int)sizeof(char), (int)sizeof(char)};
    int s3_aligns[] = {(int)alignof(char), (int)alignof(char), (int)alignof(char), (int)alignof(char)};

    int s4_sizes[]  = {(int)sizeof(char), (int)sizeof(long long), (int)sizeof(char)};
    int s4_aligns[] = {(int)alignof(char), (int)alignof(long long), (int)alignof(char)};

    int s5_sizes[]  = {(int)sizeof(int), (int)sizeof(char), (int)sizeof(int), (int)sizeof(char), (int)sizeof(short)};
    int s5_aligns[] = {(int)alignof(int), (int)alignof(char), (int)alignof(int), (int)alignof(char), (int)alignof(short)};

    const long counts[5] = {10, 25, 100, 7, 40};

    long got1 = total_padding_bytes(s1_sizes, s1_aligns, 4, counts[0]);
    long got2 = total_padding_bytes(s2_sizes, s2_aligns, 4, counts[1]);
    long got3 = total_padding_bytes(s3_sizes, s3_aligns, 4, counts[2]);
    long got4 = total_padding_bytes(s4_sizes, s4_aligns, 3, counts[3]);
    long got5 = total_padding_bytes(s5_sizes, s5_aligns, 5, counts[4]);

    printf("s1=%ld s2=%ld s3=%ld s4=%ld s5=%ld\n", got1, got2, got3, got4, got5);

    // Diagnostic only (stderr, not graded): real compiler ground truth
    // for whoever's implementation is linked in, to sanity-check ref.cpp
    // during authoring. count * (sizeof(struct) - sum of field sizes).
    auto sum = [](const int* a, int n) { long s = 0; for (int i = 0; i < n; i++) s += a[i]; return s; };
    long gt1 = ((long)sizeof(S1) - sum(s1_sizes, 4)) * counts[0];
    long gt2 = ((long)sizeof(S2) - sum(s2_sizes, 4)) * counts[1];
    long gt3 = ((long)sizeof(S3) - sum(s3_sizes, 4)) * counts[2];
    long gt4 = ((long)sizeof(S4) - sum(s4_sizes, 3)) * counts[3];
    long gt5 = ((long)sizeof(S5) - sum(s5_sizes, 5)) * counts[4];
    fprintf(stderr, "gt: s1=%ld s2=%ld s3=%ld s4=%ld s5=%ld\n", gt1, gt2, gt3, gt4, gt5);

    return 0;
}
