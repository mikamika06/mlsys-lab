import re


def reconstruct_tile_config(error_message: str) -> dict:
    match_bm = re.search(r"BLOCK_M[=:]\s*(\d+)", error_message)
    match_bn = re.search(r"BLOCK_N[=:]\s*(\d+)", error_message)
    match_bk = re.search(r"BLOCK_K[=:]\s*(\d+)", error_message)
    match_stages = re.search(r"num_stages[=:]\s*(\d+)", error_message)

    return {
        "BLOCK_M": int(match_bm.group(1)) if match_bm else 64,
        "BLOCK_N": int(match_bn.group(1)) if match_bn else 64,
        "BLOCK_K": int(match_bk.group(1)) if match_bk else 32,
        "num_stages": int(match_stages.group(1)) if match_stages else 3,
    }
