import re

def extract_triton_metadata(dump_str):
    bs_match = re.search(r"BLOCK_SIZE\s*[:=]\s*(\d+)", dump_str)
    nw_match = re.search(r"num_warps\s*[:=]\s*(\d+)", dump_str)
    grid_match = re.search(r"grid\s*[:=]\s*\[\s*(\d+)", dump_str)
    block_size = int(bs_match.group(1)) if bs_match else 0
    num_warps = int(nw_match.group(1)) if nw_match else 0
    grid = int(grid_match.group(1)) if grid_match else 0
    return {
        "block_size": block_size,
        "num_warps": num_warps,
        "grid": grid,
    }
