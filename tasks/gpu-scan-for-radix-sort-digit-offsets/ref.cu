// Reference: one radix-sort pass by `digits[i]` (already-extracted
// digit values in [0, num_digits)), done in three phases against a
// fixed-size (16-bucket) shared-memory histogram:
//   1. Histogram: count how many elements have each digit value.
//   2. Exclusive scan: turn counts into starting offsets -- offsets[d]
//      is where digit d's block of output begins (offsets[0] = 0,
//      offsets[d] = offsets[d-1] + hist[d-1]).
//   3. Stable scatter: walk the input in order; each element goes to
//      offsets[its digit] + (how many elements with that same digit
//      have already been placed), which is tracked by reusing `hist`
//      as a running per-digit cursor, incremented after each placement.
// Walking the input in its ORIGINAL order during the scatter phase is
// what makes this stable: two elements with the same digit keep their
// relative order in the output.
__global__ void radix_scatter(const float* keys, const float* digits, float* out,
                               int n, int num_digits) {
    __shared__ float hist[16];
    __shared__ float offsets[16];

    int d = 0;
    while (d < num_digits) {
        hist[d] = 0.0;
        d = d + 1;
    }

    int i = 0;
    while (i < n) {
        int dg = digits[i];
        hist[dg] = hist[dg] + 1.0;
        i = i + 1;
    }

    offsets[0] = 0.0;
    d = 1;
    while (d < num_digits) {
        offsets[d] = offsets[d - 1] + hist[d - 1];
        d = d + 1;
    }

    d = 0;
    while (d < num_digits) {
        hist[d] = 0.0;
        d = d + 1;
    }

    i = 0;
    while (i < n) {
        int dg = digits[i];
        int pos = offsets[dg] + hist[dg];
        out[pos] = keys[i];
        hist[dg] = hist[dg] + 1.0;
        i = i + 1;
    }
}
