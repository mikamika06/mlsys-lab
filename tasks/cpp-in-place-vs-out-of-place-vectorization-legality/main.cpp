// FIXED driver. Builds the 10 kernel scenarios from task.md as REAL C++
// structs and reads their layout with the real `sizeof`/`offsetof` -- never
// a simulated ABI table -- then asks the candidate's is_safe_to_vectorize
// for each and prints the verdicts.
#include <cstddef>
#include <cstdio>

#include "sol.hpp"

struct SA { int f0; double f1; };            // scenarios 0, 1, 2
struct SB { double f0; };                    // scenarios 3, 4
struct SC { char f0; int f1; double f2; };   // scenario 5
struct SD { short f0; long f1; };            // scenario 6
struct SE { float f0; float f1; };           // scenario 7
struct SF { long f0; char f1; };             // scenario 8
struct SG { int f0; int f1; double f2; };    // scenario 9

int main() {
    KernelSpec specs[10];

    // 0: in-place, same field (int) -- exempt regardless of the block math.
    specs[0] = {(int)sizeof(SA), (int)offsetof(SA, f0), (int)sizeof(int),
                (int)offsetof(SA, f0), (int)sizeof(int), 0, 0, false, 4};
    // 1: distinct fields (int -> double), but restrict-qualified -- exempt.
    specs[1] = {(int)sizeof(SA), (int)offsetof(SA, f0), (int)sizeof(int),
                (int)offsetof(SA, f1), (int)sizeof(double), 0, 0, true, 4};
    // 2: same as 1 but no restrict -- must go through the real hazard check.
    specs[2] = {(int)sizeof(SA), (int)offsetof(SA, f0), (int)sizeof(int),
                (int)offsetof(SA, f1), (int)sizeof(double), 0, 0, false, 4};
    // 3: single-field record, dest shifted +1 element (dest[i+1] = src[i]).
    specs[3] = {(int)sizeof(SB), (int)offsetof(SB, f0), (int)sizeof(double),
                (int)offsetof(SB, f0), (int)sizeof(double), 0, 1, false, 4};
    // 4: dest shifted -1 element (dest[i-1] = src[i]).
    specs[4] = {(int)sizeof(SB), (int)offsetof(SB, f0), (int)sizeof(double),
                (int)offsetof(SB, f0), (int)sizeof(double), 0, -1, false, 4};
    // 5: in-place, same field (int), but with V = 8 -- still exempt.
    specs[5] = {(int)sizeof(SC), (int)offsetof(SC, f1), (int)sizeof(int),
                (int)offsetof(SC, f1), (int)sizeof(int), 0, 0, false, 8};
    // 6: dest shifted +2 elements of a 2-byte field.
    specs[6] = {(int)sizeof(SD), (int)offsetof(SD, f0), (int)sizeof(short),
                (int)offsetof(SD, f0), (int)sizeof(short), 0, 2, false, 4};
    // 7: distinct fields (float -> float), restrict-qualified -- exempt.
    specs[7] = {(int)sizeof(SE), (int)offsetof(SE, f0), (int)sizeof(float),
                (int)offsetof(SE, f1), (int)sizeof(float), 0, 0, true, 4};
    // 8: src shifted +1 element of an 8-byte field, dest unshifted.
    specs[8] = {(int)sizeof(SF), (int)offsetof(SF, f0), (int)sizeof(long),
                (int)offsetof(SF, f0), (int)sizeof(long), 1, 0, false, 4};
    // 9: distinct int fields, no shift.
    specs[9] = {(int)sizeof(SG), (int)offsetof(SG, f0), (int)sizeof(int),
                (int)offsetof(SG, f1), (int)sizeof(int), 0, 0, false, 4};

    for (int i = 0; i < 10; i++) {
        printf("%d %d\n", i, is_safe_to_vectorize(specs[i]) ? 1 : 0);
    }
    return 0;
}
