#include <cstdio>
#include <cstddef>
#include "sol.hpp"

// Four real structs with different padding profiles. sizeof() below is
// whatever the real compiler decides -- not reimplemented anywhere.
struct A  { char a; int b; double c; };   // padding: char->int gap, int->double gap
struct Bs { float x; float y; float z; }; // no padding either way
struct Cs { double a; char b; };          // padding at the tail (round up to align 8)
struct Ds { int a; float b; };            // no padding

static void scenario(const char* tag, int struct_bytes, const int* fb, int nf,
                      const int* reads, int nr, const int* writes, int nw,
                      int flops, bool is_aos) {
    double ai = arithmetic_intensity(struct_bytes, fb, nf, reads, nr, writes, nw, flops, is_aos);
    printf("%s %.10f\n", tag, ai);
}

int main() {
    int fbA[3] = {(int)sizeof(char), (int)sizeof(int), (int)sizeof(double)};
    int sbA = (int)sizeof(A);

    int fbB[3] = {(int)sizeof(float), (int)sizeof(float), (int)sizeof(float)};
    int sbB = (int)sizeof(Bs);

    int fbC[2] = {(int)sizeof(double), (int)sizeof(char)};
    int sbC = (int)sizeof(Cs);

    int fbD[2] = {(int)sizeof(int), (int)sizeof(float)};
    int sbD = (int)sizeof(Ds);

    // 1/2: same touch pattern as the worked example -- read fields 0,1, write field 2
    {
        int r[2] = {0, 1}; int w[1] = {2};
        scenario("aos_readci_writeD", sbA, fbA, 3, r, 2, w, 1, 12, true);
        scenario("soa_readci_writeD", sbA, fbA, 3, r, 2, w, 1, 12, false);
    }

    // 3/4: read a single narrow field -- AoS drags the whole struct in, SoA doesn't
    {
        int r[1] = {1};
        scenario("aos_readb_only", sbA, fbA, 3, r, 1, nullptr, 0, 8, true);
        scenario("soa_readb_only", sbA, fbA, 3, r, 1, nullptr, 0, 8, false);
    }

    // 5/6: struct with zero padding -- AoS and SoA must agree exactly
    {
        int r[3] = {0, 1, 2};
        scenario("aos_readxyz", sbB, fbB, 3, r, 3, nullptr, 0, 6, true);
        scenario("soa_readxyz", sbB, fbB, 3, r, 3, nullptr, 0, 6, false);
    }

    // 7/8: read AND write both fields (two separate passes: one in, one out)
    {
        int r[2] = {0, 1}; int w[2] = {0, 1};
        scenario("aos_rw_both", sbC, fbC, 2, r, 2, w, 2, 20, true);
        scenario("soa_rw_both", sbC, fbC, 2, r, 2, w, 2, 20, false);
    }

    // 9/10: write-only, single narrow field
    {
        int w[1] = {1};
        scenario("aos_write_only", sbD, fbD, 2, nullptr, 0, w, 1, 4, true);
        scenario("soa_write_only", sbD, fbD, 2, nullptr, 0, w, 1, 4, false);
    }

    return 0;
}
