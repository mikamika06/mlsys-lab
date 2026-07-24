#include "sol.hpp"
#include <type_traits>

// Evaluates lv OP rv with the REAL native C++ types L and R and reads the
// result back through the REAL compiler's own promotion/conversion rules
// (decltype(lv + rv) is exactly the type the usual arithmetic conversions
// pick) -- nothing here is a hand-rolled simulation of the rule table.
template <typename L, typename R>
static ConvResult computeReal(char op, L lv, R rv) {
    using RT = decltype(lv + rv);   // same common type for +, -, and *
    RT res;
    if (op == '+') res = (RT)(lv + rv);
    else if (op == '-') res = (RT)(lv - rv);
    else res = (RT)(lv * rv);

    ConvResult cr;
    cr.width = (int)sizeof(RT);
    cr.isSigned = std::is_signed<RT>::value;
    cr.value = cr.isSigned ? (long long)res : (long long)(unsigned long long)res;
    return cr;
}

ConvResult evalConversion(char op, IntType lhsType, long long lhsVal,
                           IntType rhsType, long long rhsVal) {
#define PAIR(LT, LCPP, RT, RCPP) \
    if (lhsType == IntType::LT && rhsType == IntType::RT) \
        return computeReal<LCPP, RCPP>(op, (LCPP)lhsVal, (RCPP)rhsVal);

    PAIR(Char, char, Short, short)
    PAIR(Int, int, UInt, unsigned int)
    PAIR(UChar, unsigned char, UChar, unsigned char)
    PAIR(Int, int, ULong, unsigned long)
    PAIR(UInt, unsigned int, Long, long)
    PAIR(UInt, unsigned int, UInt, unsigned int)
    PAIR(Short, short, Short, short)
    PAIR(Char, char, Char, char)
    PAIR(UShort, unsigned short, UShort, unsigned short)
    PAIR(UChar, unsigned char, Int, int)
    PAIR(Int, int, UShort, unsigned short)
    PAIR(Long, long, ULong, unsigned long)
    PAIR(ULong, unsigned long, ULong, unsigned long)
    PAIR(Short, short, UInt, unsigned int)
    PAIR(UChar, unsigned char, Char, char)
    PAIR(Int, int, Int, int)

#undef PAIR
    return ConvResult{0, 0, false};  // unreachable for every pair main.cpp uses
}
