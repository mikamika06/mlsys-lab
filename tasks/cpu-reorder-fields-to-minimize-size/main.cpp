#include <cstdio>
#include "sol.hpp"

int main() {
    printf("naive=%zu packed=%zu\n", sizeof(NaiveStruct), packed_struct_size());
    return 0;
}
