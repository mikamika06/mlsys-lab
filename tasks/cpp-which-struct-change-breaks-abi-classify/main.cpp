#include <cstdio>
#include <cstddef>
#include <type_traits>
#include "sol.hpp"

// Byte offset of member `field` inside object `obj` (works for polymorphic
// types too, unlike offsetof on non-standard-layout classes).
#define OFF(obj, field) ((int)((char*)&(obj).field - (char*)&(obj)))

// Fill the version-independent part of a Layout straight from the compiler.
template <class T>
static Layout base() {
    Layout L{};
    L.size    = (int)sizeof(T);
    L.align   = (int)alignof(T);
    L.vptr    = std::is_polymorphic<T>::value ? 1 : 0;
    L.nfields = 0;
    return L;
}

int main() {
    const int E = 12;
    int bits[E] = {0};

    // Each scenario defines the OLD and NEW version of a struct/class, measures
    // both real layouts with the compiler, and asks the learner to classify the
    // edit. The learner never sees the definitions -- only the Layout numbers.

    // 1) Append a field at the end -> sizeof grows. BREAK.
    {
        struct O { int a; int b; };
        struct N { int a; int b; int c; };
        O o; N n;
        Layout lo = base<O>(), ln = base<N>();
        lo.nfields = ln.nfields = 2;
        lo.off[0] = OFF(o, a); lo.off[1] = OFF(o, b);
        ln.off[0] = OFF(n, a); ln.off[1] = OFF(n, b);
        bits[0] = abi_breaks(lo, ln);
    }

    // 2) Reorder fields of different sizes -> offsets move. BREAK.
    {
        struct O { int a; char b; };
        struct N { char b; int a; };
        O o; N n;
        Layout lo = base<O>(), ln = base<N>();
        lo.nfields = ln.nfields = 2;
        lo.off[0] = OFF(o, a); lo.off[1] = OFF(o, b);
        ln.off[0] = OFF(n, a); ln.off[1] = OFF(n, b);
        bits[1] = abi_breaks(lo, ln);
    }

    // 3) Change a field's type to a same-size/same-align type (int -> unsigned).
    //    Object layout is byte-identical. COMPATIBLE.
    {
        struct O { int a; int b; };
        struct N { unsigned int a; int b; };
        O o; N n;
        Layout lo = base<O>(), ln = base<N>();
        lo.nfields = ln.nfields = 2;
        lo.off[0] = OFF(o, a); lo.off[1] = OFF(o, b);
        ln.off[0] = OFF(n, a); ln.off[1] = OFF(n, b);
        bits[2] = abi_breaks(lo, ln);
    }

    // 4) Widen a field (int32 -> int64) -> size/align grow, later offset moves. BREAK.
    {
        struct O { int a; int b; };
        struct N { long long a; int b; };
        O o; N n;
        Layout lo = base<O>(), ln = base<N>();
        lo.nfields = ln.nfields = 2;
        lo.off[0] = OFF(o, a); lo.off[1] = OFF(o, b);
        ln.off[0] = OFF(n, a); ln.off[1] = OFF(n, b);
        bits[3] = abi_breaks(lo, ln);
    }

    // 5) Add a virtual to a NON-polymorphic class -> gains a vptr. BREAK.
    {
        struct O { int a; };
        struct N { virtual void f() {} int a; };
        O o; N n;
        Layout lo = base<O>(), ln = base<N>();
        lo.nfields = ln.nfields = 1;
        lo.off[0] = OFF(o, a);
        ln.off[0] = OFF(n, a);
        bits[4] = abi_breaks(lo, ln);
    }

    // 6) Append a virtual to an ALREADY-polymorphic class -> vtable grows but the
    //    object layout is unchanged. COMPATIBLE.
    {
        struct O { virtual void f() {} int a; };
        struct N { virtual void f() {} virtual void g() {} int a; };
        O o; N n;
        Layout lo = base<O>(), ln = base<N>();
        lo.nfields = ln.nfields = 1;
        lo.off[0] = OFF(o, a);
        ln.off[0] = OFF(n, a);
        bits[5] = abi_breaks(lo, ln);
    }

    // 7) Insert a field in the middle -> later field offset moves. BREAK.
    {
        struct O { int a; int c; };
        struct N { int a; int b; int c; };
        O o; N n;
        Layout lo = base<O>(), ln = base<N>();
        lo.nfields = ln.nfields = 2;               // common fields: a, c
        lo.off[0] = OFF(o, a); lo.off[1] = OFF(o, c);
        ln.off[0] = OFF(n, a); ln.off[1] = OFF(n, c);
        bits[6] = abi_breaks(lo, ln);
    }

    // 8) Reorder to reduce padding -> size shrinks and an offset moves. BREAK.
    {
        struct O { char a; int b; char c; };
        struct N { char a; char c; int b; };
        O o; N n;
        Layout lo = base<O>(), ln = base<N>();
        lo.nfields = ln.nfields = 3;               // common order: a, b, c
        lo.off[0] = OFF(o, a); lo.off[1] = OFF(o, b); lo.off[2] = OFF(o, c);
        ln.off[0] = OFF(n, a); ln.off[1] = OFF(n, b); ln.off[2] = OFF(n, c);
        bits[7] = abi_breaks(lo, ln);
    }

    // 9) Rename a field only (identical layout). COMPATIBLE.
    {
        struct O { int a; int b; };
        struct N { int a; int b; };
        O o; N n;
        Layout lo = base<O>(), ln = base<N>();
        lo.nfields = ln.nfields = 2;
        lo.off[0] = OFF(o, a); lo.off[1] = OFF(o, b);
        ln.off[0] = OFF(n, a); ln.off[1] = OFF(n, b);
        bits[8] = abi_breaks(lo, ln);
    }

    // 10) Change first field to a wider/stricter type (int -> double). BREAK.
    {
        struct O { int a; int b; };
        struct N { double a; int b; };
        O o; N n;
        Layout lo = base<O>(), ln = base<N>();
        lo.nfields = ln.nfields = 2;
        lo.off[0] = OFF(o, a); lo.off[1] = OFF(o, b);
        ln.off[0] = OFF(n, a); ln.off[1] = OFF(n, b);
        bits[9] = abi_breaks(lo, ln);
    }

    // 11) Swap two SAME-size fields -> size/align unchanged but named offsets swap. BREAK.
    {
        struct O { int a; int b; };
        struct N { int b; int a; };
        O o; N n;
        Layout lo = base<O>(), ln = base<N>();
        lo.nfields = ln.nfields = 2;               // common order: a, b
        lo.off[0] = OFF(o, a); lo.off[1] = OFF(o, b);
        ln.off[0] = OFF(n, a); ln.off[1] = OFF(n, b);
        bits[10] = abi_breaks(lo, ln);
    }

    // 12) Add a field that fits inside existing tail padding -> sizeof and every
    //     existing offset unchanged. COMPATIBLE.
    {
        struct O { int a; char b; };
        struct N { int a; char b; char c; };
        O o; N n;
        Layout lo = base<O>(), ln = base<N>();
        lo.nfields = ln.nfields = 2;               // common fields: a, b
        lo.off[0] = OFF(o, a); lo.off[1] = OFF(o, b);
        ln.off[0] = OFF(n, a); ln.off[1] = OFF(n, b);
        bits[11] = abi_breaks(lo, ln);
    }

    printf("bits:");
    int code = 0;
    for (int i = 0; i < E; i++) { printf(" %d", bits[i]); code = code * 2 + bits[i]; }
    printf("\ncode=%d\n", code);
    return 0;
}
