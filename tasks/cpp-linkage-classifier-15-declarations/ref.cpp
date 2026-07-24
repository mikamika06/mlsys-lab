#include "sol.hpp"

std::pair<std::vector<std::string>, long> classify_linkage() {
    std::vector<std::string> labels = {
        "external",  //  1. int d1;
        "internal",  //  2. static int d2;
        "internal",  //  3. const int d3 = 5;
        "external",  //  4. extern const int d4 = 5;
        "external",  //  5. void d5();
        "internal",  //  6. static void d6();
        "external",  //  7. inline void d7() {}
        "external",  //  8. extern int d8;
        "external",  //  9. class C { static int d9; };
        "none",      // 10. void f() { int d10; }
        "none",      // 11. void f() { static int d11; }
        "internal",  // 12. constexpr int d12 = 10;
        "external",  // 13. extern "C" void d13();
        "external",  // 14. const int* d14 = nullptr;   (pointer itself not const)
        "internal",  // 15. int* const d15 = nullptr;   (pointer itself const)
    };
    return {labels, (long)sizeof(S)};
}
