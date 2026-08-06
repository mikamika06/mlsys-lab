import ref


def check(workdir):
    from kvquant.calc import compute_kv_bytes

    total_err = 0.0
    for cfg in ref.CONFIGS:
        total_elements = cfg["num_tokens"] * 2 * cfg["num_layers"] * cfg["num_kv_heads"] * cfg["head_dim"]
        num_blocks = total_elements // 32
        if cfg["dtype"] == "f16":
            want = total_elements * 2
        elif cfg["dtype"] == "q8_0":
            want = num_blocks * 34
        elif cfg["dtype"] == "q4_0":
            want = num_blocks * 18

        got = compute_kv_bytes(
            cfg["num_tokens"],
            cfg["num_layers"],
            cfg["num_kv_heads"],
            cfg["head_dim"],
            cfg["dtype"],
        )
        err = abs(got - want) / float(want)
        total_err += err

    mean_err = total_err / len(ref.CONFIGS)
    return {"calc_rel_err": float(mean_err)}
