import ref


def check(workdir):
    from attnlab.traffic import estimate_traffic

    out = {"traffic_matched": 0.0, "flash_saves_bandwidth": 0.0}
    cfg = ref.CONFIGS[0]
    b, h, s, d = cfg["batch_size"], cfg["num_heads"], cfg["seq_len"], cfg["head_dim"]

    math_traffic_ref = ref.estimate_hbm_traffic(b, h, s, d, mode="math")
    flash_traffic_ref = ref.estimate_hbm_traffic(b, h, s, d, mode="flash")

    math_traffic_got = estimate_traffic(b, h, s, d, mode="math")
    flash_traffic_got = estimate_traffic(b, h, s, d, mode="flash")

    if abs(math_traffic_got - math_traffic_ref) < 1e-5 and abs(flash_traffic_got - flash_traffic_ref) < 1e-5:
        out["traffic_matched"] = 1.0

    if flash_traffic_got < math_traffic_got:
        out["flash_saves_bandwidth"] = 1.0
    else:
        out["_note"] = f"flash traffic {flash_traffic_got} not less than math traffic {math_traffic_got}"

    return out
