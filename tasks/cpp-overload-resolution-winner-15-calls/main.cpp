// FIXED driver. Each of the 15 scenarios below is REAL C++: two overloads
// tagged 0/1, and one real call expression. Ground truth is never
// hardcoded -- it is whatever the real compiler's overload resolution
// actually picks when this file is compiled, obtained by simply calling
// each scenario's winner() function and reporting the tag it returns.
#include <cstdio>
#include <string>

#include "sol.hpp"

namespace {

// 1. exact match (int) vs conversion (double), call with an int literal.
namespace s1 {
int pick(int) { return 0; }
int pick(double) { return 1; }
int winner() { return pick(5); }
}

// 2. promotion (short) vs exact (int), call with an int literal.
namespace s2 {
int pick(short) { return 0; }
int pick(int) { return 1; }
int winner() { return pick(5); }
}

// 3. non-const ref vs const ref, called with a non-const lvalue.
namespace s3 {
int pick(int&) { return 0; }
int pick(const int&) { return 1; }
int winner() { int x = 7; return pick(x); }
}

// 4. const ref vs rvalue ref, called with an rvalue.
namespace s4 {
int pick(const int&) { return 0; }
int pick(int&&) { return 1; }
int winner() { return pick(5); }
}

// 5. template vs non-template, equally good match.
namespace s5 {
template <class T> int pick(T) { return 0; }
int pick(int) { return 1; }
int winner() { return pick(5); }
}

// 6. base-ref vs derived-ref, called with a Derived object.
namespace s6 {
struct Base {};
struct Derived : Base {};
int pick(Base&) { return 0; }
int pick(Derived&) { return 1; }
int winner() { Derived d; return pick(d); }
}

// 7. void* vs bool, called with nullptr.
namespace s7 {
int pick(void*) { return 0; }
int pick(bool) { return 1; }
int winner() { return pick(nullptr); }
}

// 8. int vs ellipsis, called with an int.
namespace s8 {
int pick(int) { return 0; }
int pick(...) { return 1; }
int winner() { return pick(5); }
}

// 9. int vs double, called with an object having a user-defined
//    conversion operator to int.
namespace s9 {
struct Wrapper { operator int() const { return 42; } };
int pick(int) { return 0; }
int pick(double) { return 1; }
int winner() { Wrapper w; return pick(w); }
}

// 10. const member fn (tag 0) vs non-const member fn (tag 1), called on a
//     non-const object.
namespace s10 {
struct S {
    int pick() const { return 0; }
    int pick() { return 1; }
};
int winner() { S obj; return obj.pick(); }
}

// 11. same overload pair as scenario 10, called on a const object.
namespace s11 {
struct S {
    int pick() const { return 0; }
    int pick() { return 1; }
};
int winner() { const S obj{}; return obj.pick(); }
}

// 12. non-template pointer overload (tag 0) vs template reference-to-array
//     overload (tag 1), called with a real array passed by name.
namespace s12 {
int pick(int*) { return 0; }
template <class T> int pick(T&) { return 1; }
int winner() { int arr[5] = {1, 2, 3, 4, 5}; return pick(arr); }
}

// 13. forwarding-reference template (tag 0) vs const std::string& overload
//     (tag 1), called with a non-const lvalue std::string.
namespace s13 {
template <class T> int pick(T&&) { return 0; }
int pick(const std::string&) { return 1; }
int winner() { std::string s = "hi"; return pick(s); }
}

// 14. same overload pair as scenario 13, called with a CONST lvalue
//     std::string.
namespace s14 {
template <class T> int pick(T&&) { return 0; }
int pick(const std::string&) { return 1; }
int winner() { const std::string s = "hi"; return pick(s); }
}

// 15. char* overload (tag 0) vs const char* overload (tag 1), called with
//     a non-const char*.
namespace s15 {
int pick(char*) { return 0; }
int pick(const char*) { return 1; }
int winner() { char buf[4] = "hi"; char* p = buf; return pick(p); }
}

}  // namespace

int main() {
    int truth[15] = {
        s1::winner(),  s2::winner(),  s3::winner(),  s4::winner(),  s5::winner(),
        s6::winner(),  s7::winner(),  s8::winner(),  s9::winner(),  s10::winner(),
        s11::winner(), s12::winner(), s13::winner(), s14::winner(), s15::winner(),
    };

    int pred[15] = {};
    predict_overload_winners(pred);

    int matches = 0;
    for (int i = 0; i < 15; i++) {
        int ok = (pred[i] == truth[i]) ? 1 : 0;
        matches += ok;
        printf("%d %d\n", i + 1, ok);
    }
    printf("matches %d\n", matches);
    return 0;
}
