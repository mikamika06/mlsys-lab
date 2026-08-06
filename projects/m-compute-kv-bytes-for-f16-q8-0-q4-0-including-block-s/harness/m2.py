import ref


def check(workdir):
    from kvquant.memory import measure_kv_footprint

    total_err = 0.0
    for cfg in ref.CONFIGS:
        res = measure_kv_footprint(
            cfg["num_tokens"],
            cfg["num_layers"],
            cfg["num_kv_heads"],
            cfg["head_dim"],
            cfg["dtype"],
        )
        total_elements = cfg["num_tokens"] * 2 * cfg["num_layers"] * cfg["num_kv_heads"] * cfg["head_dim"]
        num_blocks = total_elements // 32
        if cfg["dtype"] == "f16":
            want_bytes = total_elements * 2
        elif cfg["dtype"] == "q8_0":
            want_bytes = num_blocks * 34
        elif cfg["dtype"] == "q4_0":
            want_bytes = num_blocks * 18

        got_theo = res.get("theoretical_bytes", 0)
        got_alloc = res.get("allocated_bytes", 0)

        err_theo = abs(got_theo - want_bytes) / float(want_bytes)
        err_alloc = abs(got_alloc - want_bytes) / float(want_bytes)
        total_err += (err_theo + err_alloc) / 2.0

    mean_err = total_err / len(ref.CONFIGS)
    return {"memory_rel_err": float(mean_err)}
