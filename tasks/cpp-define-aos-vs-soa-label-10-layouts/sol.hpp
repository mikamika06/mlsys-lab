#pragma once

// One field of a struct layout: its base type name and whether it's an
// array field (`type name[N]`, e.g. a whole parallel column) vs a plain
// scalar field (`type name`).
struct Field {
    const char* type;
    bool is_array;
};

// Classify a struct layout as AoS (0, "Array of Structures") or
// SoA (1, "Structure of Arrays"):
//
//   - if ANY field is an array field (a whole parallel column, e.g.
//     `float x[1000]`)                                    -> 1 (SoA)
//   - otherwise, every field is a plain per-record scalar   -> 0 (AoS)
//     (this holds even when several scalar fields happen to share the
//     same base type, e.g. a `{float r,g,b,a}` color record is still one
//     interleaved AoS record, not a column store)
//   - nfields == 0 (empty/ambiguous)                       -> 0 (AoS)
int classify_layout(const Field* fields, int nfields);
