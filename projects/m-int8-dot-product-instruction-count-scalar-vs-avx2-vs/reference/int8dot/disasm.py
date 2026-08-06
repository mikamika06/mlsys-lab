def parse_objdump_output(disasm_text: str) -> dict:
    counts = {
        "vpdpbusd": 0,
        "vpmaddubsw": 0,
        "vpmaddwd": 0,
        "vpaddd": 0,
        "total_compute": 0,
    }
    for line in disasm_text.splitlines():
        line_clean = line.strip()
        if not line_clean or ":" not in line_clean:
            continue
        parts = line_clean.split("\t")
        if len(parts) >= 3:
            instr = parts[2].strip().split()[0]
        elif len(parts) == 2:
            tokens = parts[1].strip().split()
            instr = tokens[1] if len(tokens) > 1 else tokens[0]
        else:
            continue

        if instr in counts:
            counts[instr] += 1

    counts["total_compute"] = (
        counts["vpdpbusd"]
        + counts["vpmaddubsw"]
        + counts["vpmaddwd"]
        + counts["vpaddd"]
    )
    return counts
