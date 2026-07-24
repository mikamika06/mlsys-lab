#include <cstdio>
#include <string>
#include "sol.hpp"

// Runs `nm` on this executable's own symbol table (argv[0]) and returns the
// full stdout as one string -- the REAL symbols clang++ actually emitted for
// this build, not a hand-computed prediction.
static std::string dumpSymbols(const char* exePath) {
    std::string cmd = "nm \"" + std::string(exePath) + "\" 2>/dev/null";
    std::string out;
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) return out;
    char buf[4096];
    while (fgets(buf, sizeof(buf), p)) out += buf;
    pclose(p);
    return out;
}

static bool contains(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

int main(int argc, char** argv) {
    Widget w{0};

    int a1 = w.read();   // mutable: calls 0 -> 1
    int a2 = w.read();   // mutable: calls 1 -> 2
    printf("mutable_calls %d %d\n", a1, a2);

    const Widget& cw = w;
    int c1 = cw.read();  // const: must read 2, unchanged
    int c2 = cw.read();  // const: must still read 2
    printf("const_reads %d %d\n", c1, c2);

    std::string syms = dumpSymbols(argv[0]);
    bool hasMutable = contains(syms, "_ZN6Widget4readEv");
    bool hasConst   = contains(syms, "_ZNK6Widget4readEv");
    printf("symbols_present %d %d\n", hasMutable ? 1 : 0, hasConst ? 1 : 0);

    return 0;
}
