## Context

When a header is `#include`d into several translation units (TUs), what actually happens at link time depends entirely on how each declaration is linked -- and this is a companion to `cpp-fix-an-odr-violation-across-tus`, which demonstrates the `inline`-vs-plain-external merge behavior live with a real linker on 2 real TUs. Here you predict the counts for the general case, for a header used by `numTus` TUs:

1. **`static`**: every TU emits its own *private* (internal-linkage) definition. They don't collide with each other at all -- `numTus` object-file definitions, and `numTus` separate definitions survive into the linked binary.
2. **`inline`**: every TU emits a *weak* definition -- `numTus` object-file definitions, but the linker merges them down to exactly **1** in the binary (this is the safe, ODR-compliant way to define something in a header).
3. **`extern` definition**: every TU emits a *strong* global definition -- `numTus` object-file definitions. If `numTus > 1`, the linker finds more than one strong definition of the same symbol: a real "duplicate symbol" **ODR violation**.
4. **`extern` declaration only**: no definition anywhere -- 0 object-file definitions, 0 in the binary, never an ODR violation.

## Task

Implement `predictSymbolCounts(linkage, numTus)` in `solve.cpp`, returning the object-file definition count, the linked-binary definition count, and whether it's an ODR violation, per the rules above.

## Example

```cpp
predictSymbolCounts(Linkage::Inline, 3);
// -> { objectFileDefs = 3, linkedBinaryDefs = 1, odrViolation = false }

predictSymbolCounts(Linkage::ExternDef, 2);
// -> { objectFileDefs = 2, linkedBinaryDefs = 1, odrViolation = true }
```

## What the gate checks

`main.cpp` runs 4 scenarios -- a mix of `inline`/`static`/`extern_decl` declarations across 3 TUs (including two real structs, `V1` and `V2`, whose byte sizes come from real `sizeof()`), an `extern_def` + `inline` pair across 2 TUs that must flag an ODR violation, an `inline` struct declaration across 5 TUs, and a single `extern_def` used from just 1 TU that must **not** flag a violation -- and for each, sums `objectFileDefs`/`linkedBinaryDefs` across every declared symbol, ORs the violation flags, and separately sums the real struct byte sizes of every non-`extern_decl` struct variable. Your printed totals are compared against `ref.cpp`, compiled and run the same way: `max_abs_err <= 1e-9`. Forgetting that `extern_def` only violates ODR when `numTus > 1` (scenario D) or conflating `static`'s per-TU survival with `inline`'s single-definition merge are both common mistakes this catches.
