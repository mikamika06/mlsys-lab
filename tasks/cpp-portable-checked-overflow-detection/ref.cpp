#include <climits>
#include "sol.hpp"

static bool out_of_range(__int128 r, __int128 min_val, __int128 max_val) {
    return r < min_val || r > max_val;
}

bool add_overflow_int(int a, int b) { return out_of_range((__int128)a + (__int128)b, INT_MIN, INT_MAX); }
bool sub_overflow_int(int a, int b) { return out_of_range((__int128)a - (__int128)b, INT_MIN, INT_MAX); }
bool mul_overflow_int(int a, int b) { return out_of_range((__int128)a * (__int128)b, INT_MIN, INT_MAX); }

bool add_overflow_short(short a, short b) { return out_of_range((__int128)a + (__int128)b, SHRT_MIN, SHRT_MAX); }
bool sub_overflow_short(short a, short b) { return out_of_range((__int128)a - (__int128)b, SHRT_MIN, SHRT_MAX); }
bool mul_overflow_short(short a, short b) { return out_of_range((__int128)a * (__int128)b, SHRT_MIN, SHRT_MAX); }

bool add_overflow_char(signed char a, signed char b) { return out_of_range((__int128)a + (__int128)b, SCHAR_MIN, SCHAR_MAX); }
bool sub_overflow_char(signed char a, signed char b) { return out_of_range((__int128)a - (__int128)b, SCHAR_MIN, SCHAR_MAX); }
bool mul_overflow_char(signed char a, signed char b) { return out_of_range((__int128)a * (__int128)b, SCHAR_MIN, SCHAR_MAX); }

bool add_overflow_long(long a, long b) { return out_of_range((__int128)a + (__int128)b, LONG_MIN, LONG_MAX); }
bool sub_overflow_long(long a, long b) { return out_of_range((__int128)a - (__int128)b, LONG_MIN, LONG_MAX); }
bool mul_overflow_long(long a, long b) { return out_of_range((__int128)a * (__int128)b, LONG_MIN, LONG_MAX); }
