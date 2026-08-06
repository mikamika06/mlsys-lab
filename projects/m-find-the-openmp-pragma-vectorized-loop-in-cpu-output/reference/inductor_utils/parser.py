import re

def find_cpu_vectorization(code_str):
    has_openmp = bool(re.search(r"#\s*pragma\s+omp", code_str))
    has_vectorized = bool(re.search(r"#\s*pragma\s+(?:ivdep|vector)|simd|vectorized", code_str, re.IGNORECASE))
    loop_count = len(re.findall(r"\bfor\s*\(", code_str))
    return {
        "has_openmp": has_openmp,
        "has_vectorized": has_vectorized,
        "loop_count": loop_count,
    }
