#pragma once

// Classify a UB code snippet (given as a C string containing C++ source
// text) into one of the canonical Undefined Behavior categories, using
// plain substring search (e.g. strstr) — this is a fixed-marker heuristic
// over a known snippet set, not a real static analyzer.
//
// Check IN THIS EXACT ORDER, first match wins:
//   1. contains "thread"                                -> "data-race"
//   2. contains "<<" or "2147483647"                      -> "integer"
//   3. contains "return &" or "delete" or "push_back"      -> "lifetime"
//   4. contains "(float*)" or "(short*)"                    -> "aliasing"
//   5. contains "nullptr" or "*p = 5"                        -> "null"
//   6. contains "i++"                                         -> "sequencing"
//   7. contains "arr["                                          -> "bounds"
//   8. otherwise                                                 -> "unknown"
//
// Return one of the exact string literals above (a static/string-literal
// pointer is fine — the caller only reads it, never frees it).
const char* classify_ub(const char* snippet);
