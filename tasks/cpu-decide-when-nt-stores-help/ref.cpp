#include "sol.hpp"

bool nt_stores_help(long working_set_bytes, bool reused_soon) {
    long temporal_cost = run_workload(working_set_bytes, false, reused_soon);
    long nt_cost = run_workload(working_set_bytes, true, reused_soon);
    return nt_cost < temporal_cost;
}
