// Classify each of n declared variables by memory residency: kind[i] 0
// (register-candidate), 1 (shared), 2 (global), 3 (constant). For kind
// 0, admit it as a real register (label 0) if the running total of
// register-candidate words ASSIGNED so far (spilled ones don't count)
// still fits `budget`; else it spills (label 4, spill[i]=1). Kinds
// 1/2/3 map straight to labels 1/2/3 and never touch the budget.
__global__ void classify_residency(float* label, float* spill, const float* kind,
                                    const float* size, float budget, int n) {
    // TODO: guard threadIdx.x == 0. Walk i = 0..n-1 with a running
    // register-word total, per the rule above.
}
