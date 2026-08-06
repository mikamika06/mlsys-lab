def compare_num_stages(asm2: dict, asm4: dict) -> dict:
    def count_inst(ptx):
        if not ptx: return 0
        cnt = 0
        for line in ptx.splitlines():
            line = line.strip()
            if not line: continue
            if line.startswith('.') or line.startswith('//'): continue
            if line.endswith(':'): continue
            cnt += 1
        return cnt

    p2 = asm2.get("ptx", "")
    p4 = asm4.get("ptx", "")
    return {
        "size_2": len(p2),
        "size_4": len(p4),
        "size_diff": len(p4) - len(p2),
        "inst_2": count_inst(p2),
        "inst_4": count_inst(p4),
        "inst_diff": count_inst(p4) - count_inst(p2)
    }
