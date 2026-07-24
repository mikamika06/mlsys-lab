#include "sol.hpp"

// TODO: call run_workload() once with use_nt=false and once with
// use_nt=true (same working_set_bytes, same reused_soon) and return
// true iff the non-temporal run's cost is strictly lower. See sol.hpp.
bool nt_stores_help(long working_set_bytes, bool reused_soon) {
    (void)working_set_bytes; (void)reused_soon;
    // your code here
    return false;
}
