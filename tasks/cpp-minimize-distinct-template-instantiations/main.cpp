#include <cstdio>
#include <set>
#include <string>
#include "sol.hpp"

// Runs `nm` on this executable's own symbol table (argv[0]) and returns the
// full stdout as one string -- the REAL symbols clang++ actually emitted.
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

int main(int argc, char** argv) {
    processAll();

    std::string syms = dumpSymbols(argv[0]);
    std::set<std::string> distinct;
    const std::string marker = "_Z7processI";  // mangled prefix of every process<T> instantiation
    size_t pos = 0;
    while (true) {
        size_t start = syms.find(marker, pos);
        if (start == std::string::npos) break;
        size_t end = start;
        while (end < syms.size() && syms[end] != '\n' && syms[end] != ' ' && syms[end] != '\t') end++;
        distinct.insert(syms.substr(start, end - start));
        pos = end;
    }
    printf("distinct_instantiations %d\n", (int)distinct.size());
    return 0;
}
