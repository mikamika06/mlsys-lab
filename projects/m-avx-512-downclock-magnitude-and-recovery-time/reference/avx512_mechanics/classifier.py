HEAVY_PATTERNS = (
    "vfmadd",
    "vfnmadd",
    "vpdpbusd",
    "vpdpbusds",
    "vp4dpwssd",
    "vmulps",
    "vmulpd",
    "vpmaddwd",
)


def classify_instruction(instr: str) -> str:
    """Classify an assembly instruction into ISA tier ('L0', 'L1', 'L2')."""
    s = instr.strip().lower()
    if not s or s.startswith("#") or s.startswith("//"):
        return "L0"
    if "zmm" not in s:
        return "L0"
    parts = s.split()
    opcode = parts[0] if parts else ""
    if any(pat in opcode for pat in HEAVY_PATTERNS):
        return "L2"
    return "L1"


def classify_snippet(instructions: list[str]) -> str:
    """Return the highest ISA tier present in a sequence of instructions."""
    tiers = [classify_instruction(i) for i in instructions]
    if "L2" in tiers:
        return "L2"
    if "L1" in tiers:
        return "L1"
    return "L0"
