#include "sol.hpp"

SymCounts predictSymbolCounts(Linkage linkage, int numTus) {
    SymCounts c{0, 0, false};
    switch (linkage) {
        case Linkage::Static:
            c.objectFileDefs = numTus;
            c.linkedBinaryDefs = numTus;
            break;
        case Linkage::Inline:
            c.objectFileDefs = numTus;
            c.linkedBinaryDefs = 1;
            break;
        case Linkage::ExternDef:
            c.objectFileDefs = numTus;
            c.linkedBinaryDefs = 1;
            c.odrViolation = numTus > 1;
            break;
        case Linkage::ExternDecl:
            break;
    }
    return c;
}
