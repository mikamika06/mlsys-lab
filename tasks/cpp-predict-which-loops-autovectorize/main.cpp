#include <cstdio>
#include <cstring>
#include <string>
#include "sol.hpp"

// ------------------------------------------------------------------
// The 8 fixed loops. Each is `noinline` and given "C" linkage so its
// disassembled symbol name is exactly its function name (no C++ mangling
// to strip). Whether each one autovectorizes is a real fact about this
// specific compiler/flags, determined below by disassembling this very
// executable -- never hand-simulated.
// ------------------------------------------------------------------

extern "C" __attribute__((noinline))
void vectorizable_add(float* out, const float* a, const float* b, int n) {
    for (int i = 0; i < n; i++) out[i] = a[i] + b[i];
}

extern "C" __attribute__((noinline))
void restrict_mul(float* __restrict__ out, const float* __restrict__ a, int n) {
    for (int i = 0; i < n; i++) out[i] = a[i] * 2.0f;
}

extern "C" __attribute__((noinline))
void int_add(int* out, const int* a, const int* b, int n) {
    for (int i = 0; i < n; i++) out[i] = a[i] + b[i];
}

extern "C" __attribute__((noinline))
float reduction_dep(const float* a, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] / (s + 1.0f);
    return s;
}

extern "C" __attribute__((noinline))
float plain_sum_reduction(const float* a, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i];
    return s;
}

extern "C" __attribute__((noinline))
int early_exit_loop(const float* a, int n) {
    int i;
    for (i = 0; i < n; i++) {
        if (a[i] < 0.0f) break;
    }
    return i;
}

extern "C" __attribute__((noinline)) void opaque_side_effect(float v);

extern "C" __attribute__((noinline))
void calls_opaque_fn(const float* a, int n) {
    for (int i = 0; i < n; i++) {
        opaque_side_effect(a[i]);
    }
}

extern "C" __attribute__((noinline))
void opaque_side_effect(float v) {
    static volatile float sink;
    sink = v;
}

extern "C" __attribute__((noinline))
float max_reduction(const float* a, int n) {
    float m = a[0];
    for (int i = 1; i < n; i++) if (a[i] > m) m = a[i];
    return m;
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
    float fa[64], fb[64], fout[64];
    int ia[64], ib[64], iout[64];
    for (int i = 0; i < 64; i++) {
        fa[i] = (float)(i % 7) - 3.0f;
        fb[i] = (float)(i % 5);
        ia[i] = i - 30;
        ib[i] = i * 2;
    }
    vectorizable_add(fout, fa, fb, 64);
    restrict_mul(fout, fa, 64);
    int_add(iout, ia, ib, 64);
    reduction_dep(fa, 64);
    plain_sum_reduction(fa, 64);
    early_exit_loop(fa, 64);
    calls_opaque_fn(fa, 64);
    max_reduction(fa, 64);

    std::string disasm = dumpDisassembly(argv[0]);
    const char* names[8] = {
        "vectorizable_add", "restrict_mul", "int_add", "reduction_dep",
        "plain_sum_reduction", "early_exit_loop", "calls_opaque_fn", "max_reduction",
    };
    bool actual[8];
    for (int i = 0; i < 8; i++) actual[i] = functionIsVectorized(disasm, names[i]);

    bool predicted[8] = {
        predictLoop1(), predictLoop2(), predictLoop3(), predictLoop4(),
        predictLoop5(), predictLoop6(), predictLoop7(), predictLoop8(),
    };

    for (int i = 0; i < 8; i++) {
        printf("loop%d %d %d\n", i + 1, predicted[i] ? 1 : 0, actual[i] ? 1 : 0);
    }
    return 0;
}
