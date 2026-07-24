#include <cstdio>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "sol.hpp"

namespace {
struct Rec { int access_id; int wave_id; };
std::vector<Rec> g_records;
const long INVALID_SENTINEL = 1000000000L;
}  // namespace

void schedule_access(int access_id, int wave_id) {
    g_records.push_back({access_id, wave_id});
}

long modeled_cycles() {
    if ((int)g_records.size() != TOTAL_ACCESSES) return INVALID_SENTINEL;

    std::unordered_map<int, int> wave_of;
    std::unordered_set<int> seen_ids;
    for (const auto& r : g_records) {
        if (r.access_id < 0 || r.access_id >= TOTAL_ACCESSES) return INVALID_SENTINEL;
        if (!seen_ids.insert(r.access_id).second) return INVALID_SENTINEL;  // duplicate access_id
        wave_of[r.access_id] = r.wave_id;
    }

    // Rule 1: chase steps must be strictly increasing in wave_id.
    for (int k = 0; k + 1 < NUM_CHASE_STEPS; k++) {
        if (wave_of[k + 1] <= wave_of[k]) return INVALID_SENTINEL;
    }

    // Rule 2: width limit per wave.
    std::unordered_map<int, int> width;
    for (const auto& r : g_records) width[r.wave_id]++;
    for (const auto& kv : width) {
        if (kv.second > MAX_WAVE_WIDTH) return INVALID_SENTINEL;
    }

    std::unordered_set<int> distinct_waves;
    for (const auto& r : g_records) distinct_waves.insert(r.wave_id);
    return (long)distinct_waves.size();
}

// FIXED driver. Do not edit.
int main() {
    g_records.clear();
    schedule_embedding_workload();
    printf("cycles=%ld\n", modeled_cycles());
    return 0;
}
