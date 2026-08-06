import ref


def check(workdir):
    from ggufkit import container, quants

    blob = ref.clean_blob()
    truth = ref.weights()
    index = container.parse_tensor_index(blob)
    out = {"q4k_match": 0.0, "q4k_block_match": 0.0, "half_match": 0.0}

    for bits, expect in ((0x3C00, 1.0), (0xC000, -2.0), (0x0000, 0.0),
                         (0x3555, 0.333251953125), (0x0001, 5.960464477539063e-08),
                         (0x7BFF, 65504.0)):
        if abs(quants.half_to_float(bits) - expect) > 1e-12 * max(1.0, abs(expect)):
            break
    else:
        out["half_match"] = 1.0

    for t in index["tensors"]:
        if t["ggml_type_id"] != 12:
            continue
        raw = container.tensor_bytes(blob, t)
        one = quants.dequant_q4_k(raw[:144])
        ok, _ = ref.close(one, truth[t["name"]].reshape(-1)[:256])
        out["q4k_block_match"] = 1.0 if ok else 0.0
        got = quants.dequant_tensor(raw, 12, t["n_elements"])
        ok, err = ref.close(got, truth[t["name"]])
        out["q4k_match"] = 1.0 if ok else 0.0
        out["q4k_rel_err"] = err
        break
    return out
