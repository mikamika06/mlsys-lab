#include <cstdio>
#include <cstring>
#include <string>
#include "sol.hpp"

// ------------------------------------------------------------------
// The 5 fixed loops. Each is `noinline` and given "C" linkage so its
// disassembled symbol name is exactly its function name (no C++ mangling
// to strip). Whether each one autovectorizes is a real fact about this
// specific compiler/flags, determined below by disassembling this very
// executable -- never hand-simulated.
// ------------------------------------------------------------------

extern "C" __attribute__((noinline))
void elementwise_add(float* a, const float* b, const float* c, int n) {
    for (int i = 0; i < n; i++) a[i] = b[i] + c[i];
}

extern "C" __attribute__((noinline))
void carried_dep(float* a, const float* b, int n) {
    for (int i = 1; i < n; i++) a[i] = a[i - 1] + b[i];
}

extern "C" __attribute__((noinline))
float plain_sum(const float* a, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i];
    return s;
}

extern "C" __attribute__((noinline))
void branch_free_select(float* a, const float* b, int n) {
    for (int i = 0; i < n; i++) a[i] = b[i] > 0.0f ? b[i] : 0.0f;
}

extern "C" __attribute__((noinline))
void nonuniform_index(float* a, const float* b, int n, int N) {
    for (int i = 0; i < n; i++) a[i] = b[(i * i) % N];
}

// ------------------------------------------------------------------
// Real ground truth: disassemble THIS executable's own compiled code
// (argv[0]) with `otool -tV` and check whether the named function's body
// contains a genuine NEON vector instruction (register suffix .4s/.2s/
// .2d/.16b/.8h). No simulation, no rule table -- the actual machine code.
// ------------------------------------------------------------------

static std::string dumpDisassembly(const char* exePath) {
    std::string cmd = "otool -tV \"" + std::string(exePath) + "\" 2>/dev/null";
    std::string out;
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) return out;
    char buf[4096];
    while (fgets(buf, sizeof(buf), p)) out += buf;
    pclose(p);
    return out;
}

static bool functionIsVectorized(const std::string& disasm, const char* fnName) {
    std::string label = "_" + std::string(fnName) + ":";
    size_t start = disasm.find(label);
    if (start == std::string::npos) return false;
    start = disasm.find('\n', start) + 1;
    size_t end = disasm.find("\n_", start);  // next top-level label starts a new line with '_'
    if (end == std::string::npos) end = disasm.size();
    std::string body = disasm.substr(start, end - start);
    static const char* suffixes[] = {".4s", ".2s", ".2d", ".16b", ".8h"};
    for (const char* suf : suffixes) {
        if (body.find(suf) != std::string::npos) return true;
    }
    return false;
}

int main(int argc, char** argv) {
    // exercise every loop once with real data, so nothing gets dead-code-eliminated
    float a[64], b[64], c[64];
    for (int i = 0; i < 64; i++) {
        b[i] = (float)(i % 7) - 3.0f;
        c[i] = (float)(i % 5);
    }
    elementwise_add(a, b, c, 64);
    carried_dep(a, b, 64);
    plain_sum(b, 64);
    branch_free_select(a, b, 64);
    nonuniform_index(a, b, 64, 50);

    std::string disasm = dumpDisassembly(argv[0]);
    const char* names[5] = {
        "elementwise_add", "carried_dep", "plain_sum", "branch_free_select", "nonuniform_index",
    };
    bool actual[5];
    for (int i = 0; i < 5; i++) actual[i] = functionIsVectorized(disasm, names[i]);

    bool predicted[5] = {
        predictLoop1(), predictLoop2(), predictLoop3(), predictLoop4(), predictLoop5(),
    };

    for (int i = 0; i < 5; i++) {
        printf("loop%d %d %d\n", i + 1, predicted[i] ? 1 : 0, actual[i] ? 1 : 0);
    }
    return 0;
}
