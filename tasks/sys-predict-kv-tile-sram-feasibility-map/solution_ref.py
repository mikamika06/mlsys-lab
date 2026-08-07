def kv_tile_sram_feasibility_map(configs):
    result = []
    for seq, d, sram_bytes in configs:
        required = 2 * seq * d * 2
        result.append(required <= sram_bytes)
    return result
