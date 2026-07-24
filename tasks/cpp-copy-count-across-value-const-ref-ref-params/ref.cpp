#include "sol.hpp"

void run_scenarios(int out[4]) {
    Probe obj;

    g_copy_count = 0;
    process_value(obj);
    out[0] = g_copy_count;

    g_copy_count = 0;
    process_const_ref(obj);
    out[1] = g_copy_count;

    g_copy_count = 0;
    process_ref(obj);
    out[2] = g_copy_count;

    g_copy_count = 0;
    process_value(Probe{});
    out[3] = g_copy_count;
}
