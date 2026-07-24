#include <cstdio>
#include "sol.hpp"

// FIXED driver. Walks all 10 declaration categories in a fixed order,
// classifies each one, and prints a bit vector plus a packed integer and a
// popcount so the whole answer is visible in the printed output.
int main() {
    const Category cats[10] = {
        FREE_FUNCTION, FREE_VARIABLE, CONST_VARIABLE, CLASS_TYPE,
        FUNCTION_TEMPLATE, CLASS_TEMPLATE, INLINE_VARIABLE, ENUM_TYPE,
        TYPEDEF_ALIAS, EXTERN_VARIABLE,
    };

    long long packed = 0;   // bit vector packed MSB-first into an integer
    int count = 0;          // number of categories judged "equivalent"
    for (int i = 0; i < 10; i++) {
        int bit = is_equivalent(cats[i]) ? 1 : 0;
        printf("%d ", bit);
        packed = packed * 2 + bit;
        count += bit;
    }
    printf("\n");
    printf("packed=%lld\n", packed);
    printf("count=%d\n", count);
    return 0;
}
