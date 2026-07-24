#include <cstdio>
#include "sol.hpp"

// Shown for flavor only -- sizeof is computed by the real compiler, so it
// is identical (and automatically correct) in every build; it is not part
// of what solve.cpp implements.
struct UBCategory {
    int id;
    int severity;
    const char* name;
};

int main() {
    static const char* snippets[15] = {
        "int* f() { int x = 5; return &x; }",
        "void f() { int* p = new int; delete p; int y = *p; }",
        "float f(int* p) { return *(float*)p; }",
        "short f(long* p) { return *(short*)p; }",
        "int f(int x) { return x + 2147483647; }",
        "int f(int x) { return x << 32; }",
        "int f() { int arr[5]; return arr[5]; }",
        "void f(int* arr) { arr[-1] = 0; }",
        "void f(int& x) { std::thread t([&]{ x++; }); x++; t.join(); }",
        "void f(int* p) { std::thread t([=]{ *p = 1; }); int y = *p; t.join(); }",
        "int f(int i) { return i++ + i++; }",
        "void f(int i, int* arr) { arr[i] = i++; }",
        "int f() { int* p = nullptr; return *p; }",
        "void f() { int* p = 0; *p = 5; }",
        "void f() { std::vector<int> v={1}; auto& r = v[0]; v.push_back(2); int y = r; }",
    };

    for (int i = 0; i < 15; i++) printf("%s ", classify_ub(snippets[i]));
    printf("\nsizeof(UBCategory)=%zu\n", sizeof(UBCategory));
    return 0;
}
