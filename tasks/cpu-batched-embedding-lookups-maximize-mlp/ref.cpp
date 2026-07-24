#include "sol.hpp"

// Optimal packing: one wave per chase step (the minimum forced by rule 1),
// filling every wave's remaining MAX_WAVE_WIDTH-1 slots with independent
// embedding lookups before opening a new wave.
void schedule_embedding_workload() {
    int embed_id = NUM_CHASE_STEPS;
    int per_wave_embed = MAX_WAVE_WIDTH - 1;

    for (int k = 0; k < NUM_CHASE_STEPS; k++) {
        schedule_access(k, k);
        for (int j = 0; j < per_wave_embed && embed_id < TOTAL_ACCESSES; j++, embed_id++) {
            schedule_access(embed_id, k);
        }
    }

    // Any embedding lookups left over (not needed for the fixed sizes in
    // sol.hpp, but handled generally) spill into extra waves after the
    // chase, MAX_WAVE_WIDTH per wave.
    int extra_wave = NUM_CHASE_STEPS;
    while (embed_id < TOTAL_ACCESSES) {
        int count_in_wave = 0;
        while (embed_id < TOTAL_ACCESSES && count_in_wave < MAX_WAVE_WIDTH) {
            schedule_access(embed_id, extra_wave);
            embed_id++;
            count_in_wave++;
        }
        extra_wave++;
    }
}
