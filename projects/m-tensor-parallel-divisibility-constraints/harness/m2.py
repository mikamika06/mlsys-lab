import ref


def check(workdir):
    from tp_marlin.analyze import pad_for_marlin

    out = {"padding_match": 0.0}

    for seed in [100, 101]:
        layers = ref.gen_layers(seed)
        for tp in [2, 4, 8]:
            want = ref.pad_for_marlin(layers, tp)
            got = pad_for_marlin(layers, tp)
            if want != got:
                out["_note"] = f"padding mismatch on seed {seed}, tp_size {tp}. Expected {want[:1]}, got {got[:1]}"
                return out

    out["padding_match"] = 1.0
    return out
