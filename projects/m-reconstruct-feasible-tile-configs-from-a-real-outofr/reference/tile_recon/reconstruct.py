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
