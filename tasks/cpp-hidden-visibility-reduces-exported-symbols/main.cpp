// FIXED driver. For each scenario, ground truth is computed for REAL: we
// generate an actual .cpp translation unit reproducing the declarations
// (with real `static` / `__attribute__((visibility(...)))` markup),
// compile it into a REAL shared library with the real clang++ (honoring
// -fvisibility=hidden per scenario), and count the REAL exported dynamic
// symbols with `nm -gU` (extern-only, defined-only). Never simulated.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unistd.h>

#include "sol.hpp"

namespace {

struct Decl { bool is_static; int attr; };  // attr: 0=default, 1=hidden, 2=none

int measure_exported(bool global_hidden, const Decl* decls, int n, int scenario_id) {
    std::string base = "/tmp/arena_vis_" + std::to_string((long)getpid()) + "_" + std::to_string(scenario_id);
    std::string src_path = base + ".cpp";
    std::string dylib_path = base + ".dylib";

    {
        std::ofstream out(src_path);
        for (int i = 0; i < n; i++) {
            if (decls[i].is_static) out << "static ";
            if (decls[i].attr == 0) out << "__attribute__((visibility(\"default\"))) ";
            else if (decls[i].attr == 1) out << "__attribute__((visibility(\"hidden\"))) ";
            out << "void f" << i << "() {}\n";
        }
    }

    std::string cmd = "clang++ -shared -std=c++20 ";
    if (global_hidden) cmd += "-fvisibility=hidden ";
    cmd += "-o " + dylib_path + " " + src_path + " > /dev/null 2>&1";
    std::system(cmd.c_str());

    std::string nm_cmd = "nm -gU " + dylib_path + " 2>/dev/null | wc -l";
    int count = 0;
    FILE* pipe = popen(nm_cmd.c_str(), "r");
    if (pipe) {
        char buf[64] = {};
        if (fgets(buf, sizeof(buf), pipe)) count = std::atoi(buf);
        pclose(pipe);
    }

    std::remove(src_path.c_str());
    std::remove(dylib_path.c_str());
    return count;
}

}  // namespace

int main() {
    struct Scenario {
        bool global_hidden;
        Decl decls[6];
        int n;
    };

    Scenario scenarios[] = {
        // matches the textbook example: only the explicit "default" one exports.
        {true, {{false, 2}, {true, 0}, {false, 1}, {false, 0}}, 4},
        // same declarations, but WITHOUT -fvisibility=hidden: the "none"
        // one now inherits default (exported) too.
        {false, {{false, 2}, {true, 0}, {false, 1}, {false, 0}}, 4},
        // everything static: nothing is ever exported, attributes or not.
        {true, {{true, 0}, {true, 1}, {true, 2}, {true, 0}, {true, 1}}, 5},
        // nothing static, no explicit attrs, global default (not hidden):
        // every single one exports.
        {false, {{false, 2}, {false, 2}, {false, 2}, {false, 2}, {false, 2}}, 5},
        // mixed, global hidden: only the two explicit "default" ones export.
        {true, {{false, 0}, {false, 1}, {false, 2}, {true, 0}, {false, 0}, {false, 1}}, 6},
    };
    const int NS = 5;

    int matches = 0;
    for (int s = 0; s < NS; s++) {
        int is_static[6], attr[6];
        for (int i = 0; i < scenarios[s].n; i++) {
            is_static[i] = scenarios[s].decls[i].is_static ? 1 : 0;
            attr[i] = scenarios[s].decls[i].attr;
        }
        int truth = measure_exported(scenarios[s].global_hidden, scenarios[s].decls, scenarios[s].n, s);
        int pred = count_exported_symbols(scenarios[s].global_hidden, is_static, attr, scenarios[s].n);
        int ok = (pred == truth) ? 1 : 0;
        matches += ok;
        printf("%d %d\n", s + 1, ok);
    }
    printf("matches %d\n", matches);
    return 0;
}
