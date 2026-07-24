// FIXED driver. Ground truth is never hardcoded: for each of the 15
// snippets, we write out its REAL source text, compile it for real with
// `clang++ -fsanitize=undefined -fno-sanitize-recover=all`, run the
// resulting binary, and check whether it aborted (UBSan detected the UB
// and, with -fno-sanitize-recover, terminates the process) or exited
// cleanly (no UB detected).
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unistd.h>

#include "sol.hpp"

namespace {

const char* SNIPPETS[15] = {
    // 1. signed overflow -> flagged
    "int main(){int a=2000000000,b=1500000000; volatile int c=a+b; (void)c; return 0;}",
    // 2. signed arithmetic in range -> not flagged
    "int main(){int a=100,b=100; volatile int c=a*b; (void)c; return 0;}",
    // 3. unsigned overflow (well-defined wraparound) -> not flagged
    "int main(){unsigned int a=4000000000u,b=1000000000u; volatile unsigned int c=a+b; (void)c; return 0;}",
    // 4. unsigned multiply wraparound (well-defined) -> not flagged
    "int main(){unsigned int a=3000000000u,b=2u; volatile unsigned int c=a*b; (void)c; return 0;}",
    // 5. misaligned int* dereference -> flagged
    "int main(){alignas(8) unsigned char buf[16]={}; int* p=reinterpret_cast<int*>(buf+1); volatile int v=*p; (void)v; return 0;}",
    // 6. properly aligned int* dereference -> not flagged
    "int main(){alignas(8) unsigned char buf[16]={}; int* p=reinterpret_cast<int*>(buf+4); volatile int v=*p; (void)v; return 0;}",
    // 7. misaligned double* dereference -> flagged
    "int main(){alignas(8) unsigned char buf[24]={}; double* p=reinterpret_cast<double*>(buf+1); volatile double v=*p; (void)v; return 0;}",
    // 8. shift amount >= 32 -> flagged
    "int main(){int x=1; volatile int y=x<<35; (void)y; return 0;}",
    // 9. shift amount in range -> not flagged
    "int main(){int x=1; volatile int y=x<<5; (void)y; return 0;}",
    // 10. negative shift amount -> flagged
    "int main(){int x=1, s=-1; volatile int y=x<<s; (void)y; return 0;}",
    // 11. division by zero -> flagged
    "int main(){int a=10,b=0; volatile int c=a/b; (void)c; return 0;}",
    // 12. modulo by zero -> flagged
    "int main(){int a=10,b=0; volatile int c=a%b; (void)c; return 0;}",
    // 13. division by a nonzero value -> not flagged
    "int main(){int a=10,b=5; volatile int c=a/b; (void)c; return 0;}",
    // 14. array index out of bounds -> flagged
    "int main(){int arr[5]={1,2,3,4,5}; volatile int idx=7; volatile int v=arr[idx]; (void)v; return 0;}",
    // 15. array index in bounds -> not flagged
    "int main(){int arr[5]={1,2,3,4,5}; volatile int idx=3; volatile int v=arr[idx]; (void)v; return 0;}",
};

bool measure_flagged(const char* src, int idx) {
    std::string base = "/tmp/arena_ubsan_" + std::to_string((long)getpid()) + "_" + std::to_string(idx);
    std::string src_path = base + ".cpp";
    std::string exe_path = base + ".exe";

    {
        std::ofstream out(src_path);
        out << src;
    }

    std::string compile_cmd = "clang++ -O0 -std=c++20 -fsanitize=undefined -fno-sanitize-recover=all -o " +
                               exe_path + " " + src_path + " > /dev/null 2>&1";
    int cc = std::system(compile_cmd.c_str());

    bool flagged = false;
    if (cc == 0) {
        std::string run_cmd = exe_path + " > /dev/null 2>&1";
        int rc = std::system(run_cmd.c_str());
        flagged = (rc != 0);
    }

    std::remove(src_path.c_str());
    std::remove(exe_path.c_str());
    return flagged;
}

}  // namespace

int main() {
    int truth[15];
    for (int i = 0; i < 15; i++) truth[i] = measure_flagged(SNIPPETS[i], i) ? 1 : 0;

    int pred[15] = {};
    predict_ubsan_flags(pred);

    int matches = 0;
    for (int i = 0; i < 15; i++) {
        int ok = (pred[i] == truth[i]) ? 1 : 0;
        matches += ok;
        printf("%d %d\n", i + 1, ok);
    }
    printf("matches %d\n", matches);
    return 0;
}
