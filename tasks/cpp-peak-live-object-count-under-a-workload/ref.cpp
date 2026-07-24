#include <map>
#include "sol.hpp"

void run_workload(const int* ids, const bool* is_alloc, int n) {
    std::map<int, Probe*> live;
    for (int i = 0; i < n; ++i) {
        int id = ids[i];
        if (is_alloc[i]) {
            live[id] = new Probe();
        } else {
            auto it = live.find(id);
            if (it != live.end()) {
                delete it->second;
                live.erase(it);
            }
        }
    }
    // Clean up whatever is still live at the end (doesn't affect the peak,
    // which was already latched by Probe's own constructor).
    for (auto& kv : live) {
        delete kv.second;
    }
}
