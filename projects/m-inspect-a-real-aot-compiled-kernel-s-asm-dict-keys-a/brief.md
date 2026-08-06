When debugging AOT-compiled kernels, it is crucial to analyze the generated assembly dict to track code bloat and verify hardware targets. This exercise asks you to build three analysis utilities.

1. `inspector.asm.analyze_asm_dict(asm)`: Given an `asm` dict mapping backend keys (like `"ptx"`, `"cubin"`) to strings or bytes, return a dict mapping the exact same keys to their sizes in bytes (use `len()`).
2. `inspector.stages.compare_num_stages(asm2, asm4)`: Given two `asm` dictionaries generated with `num_stages=2` and `num_stages=4`, calculate the PTX size (string length) and instruction count for each, plus the differences (`inst_diff = inst_4 - inst_2`, etc). Return a dict with keys: `"size_2"`, `"size_4"`, `"size_diff"`, `"inst_2"`, `"inst_4"`, `"inst_diff"`.
   *An instruction* is a line in the `"ptx"` string that, after `strip()`, is non-empty, does not start with `.` or `//`, and does not end with `:`. (If `"ptx"` is missing, counts and sizes are 0).
3. `inspector.portability.classify_snippet(snippet)`: Given a string, return `"CUDA"` if it contains `"sm_"`, `"ptx"`, or `".target"` (case-insensitive). Return `"ROCm"` if it contains `"amdgcn"` or `"s_waitcnt"`. Return `"Gluon"` if it contains `"gluon"`. Otherwise, return `"Unknown"`. Check in this exact order.

Finally, write a regression test in `tests/test_regression.py` that verifies your instruction counting correctly skips comments, directives, and labels. Our test runner will replace your counter with a naive `len(ptx.splitlines())`—your test must fail when it does!
