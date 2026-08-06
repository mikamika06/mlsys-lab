def analyze_throughput(isa: str, vec_bits: int, ports: int) -> dict:
    if isa == "scalar":
        elements = 1
        insts = 2
    elif isa == "avx2":
        elements = vec_bits // 8
        insts = 3
    elif isa == "avx512_vnni":
        elements = vec_bits // 8
        insts = 1
    else:
        raise ValueError(f"Unknown isa: {isa}")

    return {
        "elements_per_vec": elements,
        "instructions_per_sequence": insts,
        "macs_per_cycle": round((elements / insts) * ports, 2)
    }
