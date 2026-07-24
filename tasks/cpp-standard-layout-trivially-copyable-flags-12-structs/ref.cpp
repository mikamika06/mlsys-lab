#include "sol.hpp"
#include <type_traits>

// Correct reference: define the twelve structs exactly as documented in
// task.md and read the two ABI properties straight from <type_traits>,
// which is the oracle for standard-layout / trivially-copyable.
namespace {

struct S1 { int a; double b; };                                   // plain aggregate
struct S2 { public: int a; private: int b; };                     // mixed access control
struct S3 { int a; virtual void f(); };                           // has a virtual function
struct S4 { int a; S4() {} S4(const S4&) {} };                    // user-defined copy ctor
struct S5 { int& r; };                                            // reference data member
struct B6 { int a; }; struct S6 : B6 { int b; };                  // base AND derived hold data
struct B7 {}; struct S7 : B7 { int a; };                          // empty base, data only in derived
struct S8 { int a; ~S8() {} };                                    // user-defined destructor
struct S9 { int a; static int s; };                               // extra member is static
struct I10 { I10() {} I10(const I10&) {} };
struct S10 { I10 x; int a; };                                     // member has non-trivial copy
struct I11 { int a; int b; }; struct S11 { I11 arr[3]; int c; };  // array of standard-layout struct
struct B12 {}; struct S12 : B12 { B12 b; int a; };                // first member has same type as base

template <class T>
void put(int* out) {
    out[0] = static_cast<int>(std::is_standard_layout_v<T>);
    out[1] = static_cast<int>(std::is_trivially_copyable_v<T>);
}

} // namespace

void classify(int out[24]) {
    put<S1>(out + 0);   put<S2>(out + 2);   put<S3>(out + 4);   put<S4>(out + 6);
    put<S5>(out + 8);   put<S6>(out + 10);  put<S7>(out + 12);  put<S8>(out + 14);
    put<S9>(out + 16);  put<S10>(out + 18); put<S11>(out + 20); put<S12>(out + 22);
}
