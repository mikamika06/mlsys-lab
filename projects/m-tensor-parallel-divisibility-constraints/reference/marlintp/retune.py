from marlintp.constraints import check_marlin_eligible


def retune_marlin_config(cfg: dict) -> dict:
    candidates_n = [64, 128, 256]
    candidates_k = [64, 128]

    valid = []
    for bn in candidates_n:
        for bk in candidates_k:
            test_cfg = dict(cfg)
            test_cfg["block_n"] = bn
            test_cfg["block_k"] = bk
            if check_marlin_eligible(test_cfg):
                valid.append((bn, bk))

    if not valid:
        return {
            "eligible": False,
            "block_n": None,
            "block_k": None,
            "fallback": "decompressed_gemm",
        }

    valid.sort(key=lambda item: (item[0] * item[1], item[0]), reverse=True)
    best_bn, best_bk = valid[0]

    return {
        "eligible": True,
        "block_n": best_bn,
        "block_k": best_bk,
        "fallback": None,
    }
