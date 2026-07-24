#include "sol.hpp"

// TODO: replay stmts[0..n) as described in sol.hpp: track a stack of open
// scopes, each holding the ids/bytes of objects constructed in it; on END
// pop the current scope (normal destruction, don't record it); on THROW,
// walk every currently open scope innermost-first, and within each scope
// most-recently-constructed-first, filling out_ids and summing bytes.
long run_trace(const Stmt* stmts, int n, int* out_ids, int* out_count) {
    (void)stmts; (void)n; (void)out_ids;
    // your code here
    *out_count = 0;
    return 0;
}
