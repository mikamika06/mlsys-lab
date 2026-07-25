// Reference: classify each of n declared variables by memory residency.
// kind[i]: 0 = register-candidate scalar, 1 = __shared__, 2 = global,
// 3 = __constant__. size[i] = its size in register-words (only
// meaningful for kind 0). `budget` = the per-thread register-word
// budget. Register-candidates are assigned in ORDER: each one becomes a
// real register (label 0) if the RUNNING total of register-candidate
// words assigned SO FAR (not counting anything that already spilled)
// still fits the budget; otherwise it spills to local memory (label 4,
// spill[i] = 1) and does NOT consume any budget. shared/global/constant
// variables (labels 1/2/3) never touch the register budget at all.
__global__ void classify_residency(float* label, float* spill, const float* kind,
                                    const float* size, float budget, int n) {
    if (threadIdx.x == 0) {
        float running = 0.0f;
        for (int i = 0; i < n; i++) {
            float k = kind[i];
            if (k == 0.0f) {
                if (running + size[i] <= budget) {
                    label[i] = 0.0f;
                    spill[i] = 0.0f;
                    running = running + size[i];
                } else {
                    label[i] = 4.0f;
                    spill[i] = 1.0f;
                }
            } else if (k == 1.0f) {
                label[i] = 1.0f;
                spill[i] = 0.0f;
            } else if (k == 2.0f) {
                label[i] = 2.0f;
                spill[i] = 0.0f;
            } else {
                label[i] = 3.0f;
                spill[i] = 0.0f;
            }
        }
    }
}
