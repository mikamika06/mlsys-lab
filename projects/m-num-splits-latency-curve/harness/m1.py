import ref


def check(workdir):
    from splitkv.curve import optimal_num_splits, predict_latency

    out = {"curves_matched": 1.0, "optimal_matched": 1.0}

    for cfg in ref.CONFIGS:
        bs = cfg["batch_size"]
        sl = cfg["seq_len"]
        sm = cfg["num_sm"]

        for s in [1, 2, 4, 8, 16, 32]:
            want_lat = ref.predict_latency(bs, sl, s, sm)
            got_lat = predict_latency(bs, sl, s, sm)
            if abs(want_lat - got_lat) > 1e-4:
                out["curves_matched"] = 0.0
                out["_note"] = f"bs={bs}, sl={sl}, splits={s}: expected {want_lat}, got {got_lat}"
                break
        if out["curves_matched"] == 0.0:
            break

    for cfg in ref.CONFIGS:
        bs = cfg["batch_size"]
        sl = cfg["seq_len"]
        sm = cfg["num_sm"]

        want_opt = ref.optimal_num_splits(bs, sl, sm)
        got_opt = optimal_num_splits(bs, sl, sm)
        if want_opt != got_opt:
            out["optimal_matched"] = 0.0
            out["_note"] = f"bs={bs}, sl={sl}: expected opt splits {want_opt}, got {got_opt}"
            break

    return out
