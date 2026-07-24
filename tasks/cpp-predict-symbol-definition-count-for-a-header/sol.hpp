#pragma once

// How ONE header-declared symbol is linked, when the header is included by
// several translation units (TUs).
enum class Linkage { Static, Inline, ExternDef, ExternDecl };

struct SymCounts {
    long long objectFileDefs;    // definitions across all `numTus` object files, before linking
    long long linkedBinaryDefs;  // distinct definitions surviving into the final linked binary
    bool odrViolation;           // true if linking this hits a "duplicate symbol" ODR error
};

// Predicts the above for a symbol declared with `linkage`, used across
// `numTus` translation units:
//
//   Static:      each TU emits its OWN private (internal-linkage)
//                definition -- numTus object-file defs, and numTus SEPARATE
//                surviving definitions in the binary (they don't collide,
//                so nothing gets merged).
//   Inline:      each TU emits a weak definition -- numTus object-file
//                defs, but the linker MERGES them down to exactly 1.
//   ExternDef:   each TU emits a STRONG global definition -- numTus
//                object-file defs; if numTus > 1 this is an ODR violation
//                (a real "duplicate symbol" link error), and the surviving
//                count is 1 regardless (that's what the linker keeps, or
//                would keep, before erroring).
//   ExternDecl:  a declaration only, no definition anywhere -- 0 and 0,
//                never an ODR violation.
SymCounts predictSymbolCounts(Linkage linkage, int numTus);
