#include <cstdio>
#include "sol.hpp"

// Fixed driver: 12 deterministic C++ constructs, each described by
// (kind, linkage, is_inline). For each one it prints whether that construct may
// legally be defined in more than one translation unit, then a packed 12-bit
// value (bit i = verdict for construct i) and the total count of "yes" verdicts.
int main() {
    struct Construct {
        const char* label;
        int kind;
        int linkage;
        int is_inline;
    };
    const Construct constructs[] = {
        // 0: int add(int,int){...}                  non-inline free function
        {"free_function",        KIND_FUNCTION, LINK_EXTERNAL, 0},
        // 1: inline int sq(int){...}                inline function
        {"inline_function",      KIND_FUNCTION, LINK_EXTERNAL, 1},
        // 2: struct Point { int x, y; };            class/struct type
        {"struct_definition",    KIND_CLASS,    LINK_EXTERNAL, 0},
        // 3: template<class T> T maxv(T,T){...}     function template
        {"function_template",    KIND_TEMPLATE, LINK_EXTERNAL, 0},
        // 4: int g_counter;                         non-inline global variable
        {"global_variable",      KIND_VARIABLE, LINK_EXTERNAL, 0},
        // 5: inline int g_flag = 0;                 inline variable (C++17)
        {"inline_variable",      KIND_VARIABLE, LINK_EXTERNAL, 1},
        // 6: const int kN = 42;                     const namespace-scope var (internal)
        {"const_namespace_var",  KIND_VARIABLE, LINK_INTERNAL, 0},
        // 7: static int helper(){...}               static (internal-linkage) function
        {"static_function",      KIND_FUNCTION, LINK_INTERNAL, 0},
        // 8: enum Color { Red, Green, Blue };       enumeration type
        {"enum_definition",      KIND_ENUM,     LINK_EXTERNAL, 0},
        // 9: using Byte = unsigned char;            type alias
        {"type_alias",           KIND_ALIAS,    LINK_EXTERNAL, 0},
        //10: template<class T> struct Box {...};    class template
        {"class_template",       KIND_TEMPLATE, LINK_EXTERNAL, 0},
        //11: int Widget::area() const {...}         out-of-class member function (non-inline)
        {"member_fn_out_of_line",KIND_FUNCTION, LINK_EXTERNAL, 0},
    };
    const int N = (int)(sizeof(constructs) / sizeof(constructs[0]));

    int packed = 0;
    int count = 0;
    for (int i = 0; i < N; i++) {
        int v = may_appear_in_multiple_tus(constructs[i].kind,
                                           constructs[i].linkage,
                                           constructs[i].is_inline);
        v = v ? 1 : 0;
        printf("%d ", v);
        if (v) packed |= (1 << i);
        count += v;
    }
    printf("\npacked=%d count=%d\n", packed, count);
    return 0;
}
