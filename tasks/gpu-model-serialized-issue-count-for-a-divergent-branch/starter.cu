// TODO: if lanes_taking_then is 0, only the else path issues
// (issues = else_instrs); if it's 32, only the then path issues
// (issues = then_instrs); otherwise (a genuinely divergent warp) BOTH
// paths issue, serially: issues = then_instrs + else_instrs. Then
// penalty = issues / max(then_instrs, else_instrs). Write issues to
// out[0], penalty to out[1]. See ref.cu -- note the "+ 0.0" there is
// deliberate, forcing float arithmetic for the division.
__global__ void divergent_issue_count(int then_instrs, int else_instrs,
                                       int lanes_taking_then, float* out) {
    out[0] = 0.0;
    out[1] = 0.0;
}
