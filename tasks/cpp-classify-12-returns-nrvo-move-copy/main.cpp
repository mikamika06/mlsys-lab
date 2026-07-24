// FIXED driver. Defines the instrumented struct T and the 12 functions
// f1..f12 documented in task.md verbatim, runs each one for real, and
// classifies its actual observed behaviour from real copy/move constructor
// call counts (never a hardcoded table) -- that classification is the
// ground truth the candidate's prediction is graded against.
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>

#include "sol.hpp"

static int g_copies = 0;
static int g_moves = 0;

struct T {
    char c;
    double d;
    int i;
    T() : c(0), d(0), i(0) {}
    T(char c_, double d_, int i_) : c(c_), d(d_), i(i_) {}
    T(const T& o) : c(o.c), d(o.d), i(o.i) { g_copies++; }
    T(T&& o) noexcept : c(o.c), d(o.d), i(o.i) { g_moves++; }
    T& operator=(const T&) = default;
    T& operator=(T&&) = default;
};

// The 12 return scenarios from task.md, reproduced verbatim.
T f1() { T t; return t; }
T f2() { return T(); }
T f3(T t) { return t; }
T f4(T& t) { return t; }
T f5() { T t; return std::move(t); }
static T g_global6;
T f6() { return g_global6; }
T f7() { static T t; return t; }
T f8() { T* t = new T(); return *t; }
T f9(bool b) { T t1, t2; return b ? t1 : t2; }
T f10() { T t; return (t); }
struct U { T t; };
T f11(U u) { return u.t; }
T f12() { return T{'a', 1.0, 42}; }

// Turn observed (copies, moves) into the ground-truth label. `named_return`
// disambiguates the two ways of seeing zero extra constructor calls: the
// operand was a prvalue (rvo, guaranteed) vs. the name of a single local
// automatic object (nrvo, elision permitted -- and observed here).
static std::string classify(int copies, int moves, bool named_return) {
    if (copies == 0 && moves == 0) return named_return ? "nrvo" : "rvo";
    if (copies == 0) return "move";
    return "copy";
}

int main() {
    std::string truth[12];
    int true_size = (int)sizeof(T);

    { g_copies = 0; g_moves = 0; T r = f1(); (void)r;
      truth[0] = classify(g_copies, g_moves, /*named=*/true); }
    { g_copies = 0; g_moves = 0; T r = f2(); (void)r;
      truth[1] = classify(g_copies, g_moves, /*named=*/false); }
    { g_copies = 0; g_moves = 0; T r = f3(T()); (void)r;  // prvalue arg: isolates the return itself
      truth[2] = classify(g_copies, g_moves, /*named=*/false); }
    T p4;
    { g_copies = 0; g_moves = 0; T r = f4(p4); (void)r;
      truth[3] = classify(g_copies, g_moves, /*named=*/false); }
    { g_copies = 0; g_moves = 0; T r = f5(); (void)r;
      truth[4] = classify(g_copies, g_moves, /*named=*/false); }
    { g_copies = 0; g_moves = 0; T r = f6(); (void)r;
      truth[5] = classify(g_copies, g_moves, /*named=*/false); }
    { g_copies = 0; g_moves = 0; T r = f7(); (void)r;
      truth[6] = classify(g_copies, g_moves, /*named=*/false); }
    { g_copies = 0; g_moves = 0; T r = f8(); (void)r;
      truth[7] = classify(g_copies, g_moves, /*named=*/false); }
    { g_copies = 0; g_moves = 0; T r = f9(true); (void)r;
      truth[8] = classify(g_copies, g_moves, /*named=*/false); }
    { g_copies = 0; g_moves = 0; T r = f10(); (void)r;
      truth[9] = classify(g_copies, g_moves, /*named=*/true); }
    { g_copies = 0; g_moves = 0; T r = f11(U()); (void)r;  // prvalue arg: isolates the return itself
      truth[10] = classify(g_copies, g_moves, /*named=*/false); }
    { g_copies = 0; g_moves = 0; T r = f12(); (void)r;
      truth[11] = classify(g_copies, g_moves, /*named=*/false); }

    std::string pred[12];
    predict_return_kinds(pred);
    int pred_size = predict_struct_size();

    int matches = 0;
    for (int idx = 0; idx < 12; idx++) {
        int ok = (pred[idx] == truth[idx]) ? 1 : 0;
        matches += ok;
        printf("%d %d\n", idx + 1, ok);
    }
    printf("matches %d\n", matches);
    printf("size_pred %d\n", pred_size);
    printf("size_true %d\n", true_size);
    printf("size_ok %d\n", pred_size == true_size ? 1 : 0);
    return 0;
}
