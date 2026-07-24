#pragma once
#include <string>
#include <vector>
#include <utility>

// ---------------------------------------------------------------------------
// PROVIDED (do not change): the real, compiler-laid-out struct used for
// the struct_size half of this task.
// ---------------------------------------------------------------------------
struct S {
    double d;
    int    i;
};

// ---------------------------------------------------------------------------
// LEARNER IMPLEMENTS.
//
// Classify the linkage of each of the following 15 declarations
// (namespace scope unless noted) as "external", "internal", or "none",
// IN THIS ORDER:
//
//    1.  int d1;
//    2.  static int d2;
//    3.  const int d3 = 5;
//    4.  extern const int d4 = 5;
//    5.  void d5();
//    6.  static void d6();
//    7.  inline void d7() {}
//    8.  extern int d8;
//    9.  class C { static int d9; };               -- target: d9
//   10.  void f() { int d10; }                      -- target: d10
//   11.  void f() { static int d11; }                -- target: d11
//   12.  constexpr int d12 = 10;
//   13.  extern "C" void d13();
//   14.  const int* d14 = nullptr;
//   15.  int* const d15 = nullptr;
//
// Rules to apply:
//   - `static` at namespace scope -> internal linkage.
//   - A non-extern, non-`inline` object at namespace scope whose OWN
//     declared type is const-qualified (top-level const on the entity
//     itself, not merely "pointer to const") defaults to internal
//     linkage unless explicitly `extern`. `constexpr` implies `const`,
//     so the same default rule applies to it.
//   - A pointer variable whose pointee is const but whose own type is
//     NOT const-qualified (e.g. `const int* p`) is an ordinary,
//     non-const-qualified declaration -> default (external) linkage.
//     A pointer variable that is itself const-qualified (e.g.
//     `int* const p`) IS const-qualified -> the const-default-internal
//     rule applies to it.
//   - Ordinary (non-static, non-const) namespace-scope objects and
//     functions, `inline` functions, and `extern "C"` declarations
//     default to external linkage.
//   - A class member declared `static` has external linkage by default
//     (it needs one, since it must be defined exactly once across the
//     program).
//   - A local variable (function scope) has NO linkage at all, whether
//     or not it is itself declared `static` -- its name is never visible
//     outside the function, only its storage duration changes.
//
// Return {labels, struct_size}: labels is 15 strings in the order above,
// struct_size is sizeof(S) as the real compiler lays it out.
// ---------------------------------------------------------------------------
std::pair<std::vector<std::string>, long> classify_linkage();
