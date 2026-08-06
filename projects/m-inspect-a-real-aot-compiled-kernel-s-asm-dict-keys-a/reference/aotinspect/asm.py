def inspect_asm_dict(asm_dict: dict) -> dict:
    """Inspect assembly dictionary keys, line counts, and byte sizes."""
    out = {}
    for k, v in asm_dict.items():
        is_bin = isinstance(v, bytes)
        if is_bin:
            size = len(v)
            lines = 0
        else:
            size = len(v.encode("utf-8"))
            lines = len(v.splitlines())
        out[k] = {"byte_size": size, "is_binary": is_bin, "line_count": lines}
    return out


def compare_ptx_stages(ptx_stage_a: str, ptx_stage_b: str) -> dict:
    """Compare instruction counts and byte sizes between two PTX strings."""
    def _count_insts(ptx: str) -> int:
        count = 0
        for line in ptx.splitlines():
            s = line.strip()
            if not s or s.startswith("//") or s.startswith(".") or s.endswith(":"):
                continue
            count += 1
        return count

    a_inst = _count_insts(ptx_stage_a)
    b_inst = _count_insts(ptx_stage_b)
    a_bytes = len(ptx_stage_a.encode("utf-8"))
    b_bytes = len(ptx_stage_b.encode("utf-8"))

    return {
        "stage_a_bytes": a_bytes,
        "stage_b_bytes": b_bytes,
        "stage_a_instructions": a_inst,
        "stage_b_instructions": b_inst,
        "instruction_ratio": float(b_inst / a_inst) if a_inst > 0 else 0.0,
        "byte_ratio": float(b_bytes / a_bytes) if a_bytes > 0 else 0.0,
    }
