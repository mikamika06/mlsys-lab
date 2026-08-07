MAX_SMEM = 49152
ELEMENT_SIZE = 4
CANDIDATES = [
    {"BLOCK_M": 16, "BLOCK_N": 16, "stages": 2, "overhead": 100},
    {"BLOCK_M": 32, "BLOCK_N": 32, "stages": 2, "overhead": 100},
    {"BLOCK_M": 64, "BLOCK_N": 64, "stages": 2, "overhead": 100},
    {"BLOCK_M": 128, "BLOCK_N": 128, "stages": 4, "overhead": 100},
]
ERROR_STRINGS = [
    "OutOfResources: out of shared memory in block execution",
    "CompilationError: exceeded maximum register count per thread",
    "RuntimeError: grid size too large for device limits",
    "SyntaxError: invalid syntax in triton kernel"
]
EXPECTED_CLASSES = [
    "SHARED_MEMORY_EXCEEDED",
    "REGISTER_PRESSURE",
    "GRID_SIZE_LIMIT",
    "UNKNOWN_OR_SYNTAX"
]


def reconstruct_configs(max_smem, element_size, candidates):
    out = []
    for c in candidates:
        bm = c["BLOCK_M"]
        bn = c["BLOCK_N"]
        stages = c["stages"]
        smem = (bm * bn * element_size * stages) + c.get("overhead", 0)
        if smem <= max_smem:
            out.append(c)
    return out


def classify_errors(error_strings):
    results = []
    for err in error_strings:
        lower = err.lower()
        if "shared memory" in lower or "smem" in lower:
            results.append("SHARED_MEMORY_EXCEEDED")
        elif "register" in lower or "regs" in lower:
            results.append("REGISTER_PRESSURE")
        elif "grid" in lower or "block size" in lower:
            results.append("GRID_SIZE_LIMIT")
        else:
            results.append("UNKNOWN_OR_SYNTAX")
    return results
