#pragma once

// A category of namespace-scope declaration that can be given internal
// linkage either by the `static` keyword or by placing it inside an
// anonymous namespace.
enum Category {
    FREE_FUNCTION      = 0,  // void f() {}
    FREE_VARIABLE      = 1,  // int x;
    CONST_VARIABLE     = 2,  // const int x = 1;
    CLASS_TYPE         = 3,  // struct/class definition
    FUNCTION_TEMPLATE  = 4,  // template <class T> T f(T);
    CLASS_TEMPLATE     = 5,  // template <class T> struct S { ... };
    INLINE_VARIABLE    = 6,  // inline int x = 1;   (C++17)
    ENUM_TYPE          = 7,  // enum E { ... };
    TYPEDEF_ALIAS      = 8,  // typedef int I;  /  using I = int;
    EXTERN_VARIABLE    = 9,  // extern int x;
};

// Return 1 if declaring `c` with the `static` keyword at namespace scope
// produces the SAME observable behavior as declaring it inside an
// anonymous namespace instead (both legal, both give internal linkage, no
// ODR difference). Return 0 if the two approaches diverge for `c` (e.g.
// `static` cannot legally be applied to that category, or the resulting
// semantics differ). See task.md for the C++20 rules behind each case.
int is_equivalent(Category c);
