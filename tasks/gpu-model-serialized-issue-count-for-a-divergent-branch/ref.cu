// Reference: SIMT branch divergence model. A warp executing an
// if/else where lanes disagree must issue BOTH paths, one after the
// other, masking off whichever lanes aren't participating in each pass
// -- so a divergent warp's total issued instructions is then_instrs +
// else_instrs, always, regardless of how the 32 lanes split. Only when
// every lane agrees (lanes_taking_then is 0 or 32) does the warp skip
// the path it never takes.
__global__ void divergent_issue_count(int then_instrs, int else_instrs,
                                       int lanes_taking_then, float* out) {
    float t = then_instrs + 0.0;
    float e = else_instrs + 0.0;
    float issues;
    if (lanes_taking_then == 0) {
        issues = e;
    } else if (lanes_taking_then == 32) {
        issues = t;
    } else {
        issues = t + e;
    }
    float baseline = t > e ? t : e;
    float penalty = issues / baseline;
    out[0] = issues;
    out[1] = penalty;
}
