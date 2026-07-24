#include <cstdio>
#include "sol.hpp"

static void printResult(const char* tag, const ConvResult& r) {
    if (r.isSigned) printf("%s %lld %d 1\n", tag, r.value, r.width);
    else printf("%s %llu %d 0\n", tag, (unsigned long long)r.value, r.width);
}

int main() {
    using T = IntType;
    struct Expr { const char* tag; char op; T lt; long long lv; T rt; long long rv; };
    Expr exprs[] = {
        {"e1",  '+', T::Char,   100,        T::Short,  200},
        {"e2",  '+', T::Int,    -1,         T::UInt,   1},
        {"e3",  '+', T::UChar,  200,        T::UChar,  100},
        {"e4",  '*', T::Int,    -5,         T::ULong,  10},
        {"e5",  '+', T::UInt,   4294967295, T::Long,   5},
        {"e6",  '-', T::UInt,   100,        T::UInt,   200},
        {"e7",  '*', T::Short,  300,        T::Short,  300},
        {"e8",  '+', T::Char,   -120,       T::Char,   -20},
        {"e9",  '+', T::UShort, 60000,      T::UShort, 10000},
        {"e10", '-', T::UChar,  10,         T::Int,    20},
        {"e11", '*', T::Int,    1000,       T::UShort, 2000},
        {"e12", '+', T::Long,   -500,       T::ULong,  1000},
        {"e13", '-', T::ULong,  10,         T::ULong,  20},
        {"e14", '*', T::Short,  -10,        T::UInt,   5},
        {"e15", '+', T::UChar,  255,        T::Char,   1},
        {"e16", '*', T::Int,    -1,         T::Int,    -1},
    };
    for (const auto& e : exprs) {
        ConvResult r = evalConversion(e.op, e.lt, e.lv, e.rt, e.rv);
        printResult(e.tag, r);
    }
    return 0;
}
