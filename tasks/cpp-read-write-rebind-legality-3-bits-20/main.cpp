// FIXED driver. Ground truth is never hardcoded for the read/write bits:
// for each declaration we generate a REAL tiny .cpp program performing
// exactly that operation and ask the REAL clang++ (`-fsyntax-only`,
// `-std=c++20`) whether it compiles. A reference can never be rebound in
// C++ -- there is no syntax that even attempts it, so that one fact (and
// only that one) is stated directly rather than "tested" by a compiler
// invocation that has nothing to compile.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unistd.h>

#include "sol.hpp"

namespace {

struct Case {
    const char* decl;    // setup: declares the entity (and any helper vars)
    const char* read_op;
    const char* write_op;
    const char* rebind_op;  // "" for reference types (rebind is never legal)
    bool is_reference;
};

const Case CASES[20] = {
    {"int* p = &g;",                                                  "int v = *p; (void)v;",    "*p = 1;",   "p = &g2;",  false},
    {"const int* p = &g;",                                            "int v = *p; (void)v;",    "*p = 1;",   "p = &g2;",  false},
    {"int* const p = &g;",                                            "int v = *p; (void)v;",    "*p = 1;",   "p = &g2;",  false},
    {"const int* const p = &g;",                                      "int v = *p; (void)v;",    "*p = 1;",   "p = &g2;",  false},
    {"int& r = g;",                                                   "int v = r; (void)v;",     "r = 1;",    "",          true},
    {"const int& r = g;",                                             "int v = r; (void)v;",     "r = 1;",    "",          true},
    {"double* p = &dg;",                                              "double v = *p; (void)v;", "*p = 1.0;", "p = &dg2;", false},
    {"const double* p = &dg;",                                        "double v = *p; (void)v;", "*p = 1.0;", "p = &dg2;", false},
    {"double* const p = &dg;",                                        "double v = *p; (void)v;", "*p = 1.0;", "p = &dg2;", false},
    {"char& r = cg;",                                                 "char v = r; (void)v;",    "r = 'z';",  "",          true},
    {"const char& r = cg;",                                           "char v = r; (void)v;",    "r = 'z';",  "",          true},
    {"void* p = &g;",                                                 "auto v = *p; (void)v;",   "*p = 1;",   "p = &g2;",  false},
    {"const void* p = &g;",                                           "auto v = *p; (void)v;",   "*p = 1;",   "p = &g2;",  false},
    {"void* const p = &g;",                                           "auto v = *p; (void)v;",   "*p = 1;",   "p = &g2;",  false},
    {"int* ip15 = &g; int*& rp = ip15;",                              "int v = *rp; (void)v;",   "*rp = 1;",  "",          true},
    {"int* const ip16 = &g; int* const& rp = ip16;",                  "int v = *rp; (void)v;",   "*rp = 1;",  "",          true},
    {"const int* ip17 = &g; const int*& rp = ip17;",                  "int v = *rp; (void)v;",   "*rp = 1;",  "",          true},
    {"const int* const ip18 = &g; const int* const& rp = ip18;",      "int v = *rp; (void)v;",   "*rp = 1;",  "",          true},
    {"long* const p = &lg;",                                          "long v = *p; (void)v;",   "*p = 1;",   "p = &lg2;", false},
    {"const long* p = &lg;",                                          "long v = *p; (void)v;",   "*p = 1;",   "p = &lg2;", false},
};

bool compiles(const std::string& decl, const std::string& op, int idx, int which) {
    std::string base = "/tmp/arena_legal_" + std::to_string((long)getpid()) + "_" + std::to_string(idx) + "_" + std::to_string(which);
    std::string src_path = base + ".cpp";
    {
        std::ofstream out(src_path);
        out << "int g = 5, g2 = 6; double dg = 5.0, dg2 = 6.0; char cg = 'x'; long lg = 5L, lg2 = 6L;\n";
        out << "int main() {\n    " << decl << "\n    " << op << "\n    return 0;\n}\n";
    }
    std::string cmd = "clang++ -fsyntax-only -std=c++20 " + src_path + " > /dev/null 2>&1";
    int rc = std::system(cmd.c_str());
    std::remove(src_path.c_str());
    return rc == 0;
}

}  // namespace

int main() {
    int truth[60];
    for (int i = 0; i < 20; i++) {
        const Case& c = CASES[i];
        truth[i * 3 + 0] = compiles(c.decl, c.read_op, i, 0) ? 1 : 0;
        truth[i * 3 + 1] = compiles(c.decl, c.write_op, i, 1) ? 1 : 0;
        truth[i * 3 + 2] = c.is_reference ? 0 : (compiles(c.decl, c.rebind_op, i, 2) ? 1 : 0);
    }

    int pred[60] = {};
    predict_legality(pred);

    int matches = 0;
    for (int i = 0; i < 60; i++) {
        int ok = (pred[i] == truth[i]) ? 1 : 0;
        matches += ok;
        printf("%d %d\n", i + 1, ok);
    }
    printf("matches %d\n", matches);
    return 0;
}
