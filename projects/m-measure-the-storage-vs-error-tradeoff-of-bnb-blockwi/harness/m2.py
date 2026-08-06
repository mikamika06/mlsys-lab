import ref


def check(workdir):
    from bnbquant.evaluate import measure_tradeoff

    out = {"tradeoff_match": 0.0}
    ok = 0
    for t in ref.TENSORS:
        ref_res = ref.measure_tradeoff(t, ref.BLOCK_SIZES, ref.BITS_LIST)
        try:
            got_res = measure_tradeoff(t, ref.BLOCK_SIZES, ref.BITS_LIST)
        except Exception:
            continue

        if len(ref_res) == len(got_res):
            match = True
            for r_item, g_item in zip(ref_res, got_res):
                if r_item["block_size"] != g_item["block_size"] or r_item["bits"] != g_item["bits"]:
                    match = False
                    break
                if abs(r_item["storage_bytes"] - g_item["storage_bytes"]) > 0:
                    match = False
                    break
                if abs(r_item["mse"] - g_item["mse"]) > 1e-5:
                    match = False
                    break
            if match:
                ok += 1
    if ok == len(ref.TENSORS):
        out["tradeoff_match"] = 1.0
    return out
