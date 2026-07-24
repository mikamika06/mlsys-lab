#include <cstdio>
#include "sol.hpp"

// FIXED driver: five objects, one with a negative value (triggers the error
// path on the 4th object). Prints the return code and every object's final
// ob_refcnt.

int main() {
    long values[5] = {10, 20, 30, -5, 40};
    MyPyObject objs[5];
    for (int i = 0; i < 5; ++i) {
        objs[i].ob_refcnt = 1;
        objs[i].ob_type = nullptr;
        objs[i].value = values[i];
    }

    int rc = process_items(objs, 5);

    printf("rc=%d\n", rc);
    for (int i = 0; i < 5; ++i) {
        printf("%ld ", objs[i].ob_refcnt);
    }
    printf("\n");
    return 0;
}
