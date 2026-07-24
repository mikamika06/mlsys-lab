#include <cstdio>
#include "sol.hpp"

struct V1 { int a; double b; };   // real sizeof = 16
struct V2 { long a; char b; };    // real sizeof = 16
struct S1 { char a; int b; double c; };  // real sizeof = 16

struct Decl { const char* name; Linkage linkage; long long structBytes; };

static void runScenario(const char* tag, Decl* decls, int n, int numTus) {
    long long objTotal = 0, linkedTotal = 0, bytesPerTu = 0;
    bool odr = false;
    for (int i = 0; i < n; i++) {
        SymCounts c = predictSymbolCounts(decls[i].linkage, numTus);
        objTotal += c.objectFileDefs;
        linkedTotal += c.linkedBinaryDefs;
        odr = odr || c.odrViolation;
        if (decls[i].linkage != Linkage::ExternDecl) bytesPerTu += decls[i].structBytes;
    }
    printf("%s %lld %lld %d %lld\n", tag, objTotal, linkedTotal, odr ? 1 : 0, bytesPerTu);
}

int main() {
    // Scenario A: f1(inline,fn) f2(static,fn) v1(extern_decl,struct V1) v2(static,struct V2), 3 TUs
    {
        Decl decls[] = {
            {"f1", Linkage::Inline, 0},
            {"f2", Linkage::Static, 0},
            {"v1", Linkage::ExternDecl, (long long)sizeof(V1)},
            {"v2", Linkage::Static, (long long)sizeof(V2)},
        };
        runScenario("scenarioA", decls, 4, 3);
    }

    // Scenario B: fn_ext(extern_def,fn) fn_inl(inline,fn), 2 TUs -- triggers an ODR violation
    {
        Decl decls[] = {
            {"fn_ext", Linkage::ExternDef, 0},
            {"fn_inl", Linkage::Inline, 0},
        };
        runScenario("scenarioB", decls, 2, 2);
    }

    // Scenario C: s1(inline,struct S1) s2(extern_decl, no struct), 5 TUs
    {
        Decl decls[] = {
            {"s1", Linkage::Inline, (long long)sizeof(S1)},
            {"s2", Linkage::ExternDecl, 0},
        };
        runScenario("scenarioC", decls, 2, 5);
    }

    // Scenario D: a lone extern_def used from just 1 TU -- must NOT be an ODR violation
    {
        Decl decls[] = {
            {"solo", Linkage::ExternDef, 0},
        };
        runScenario("scenarioD", decls, 1, 1);
    }

    return 0;
}
