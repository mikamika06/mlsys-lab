import ref


def check(workdir):
    from ggufkit import container, quants

    blob = ref.clean_blob()
    truth = ref.weights()
    index = container.parse_tensor_index(blob)
    out = {"q6k_match": 0.0, "q6k_block_match": 0.0, "f32_match": 0.0}

    for t in index["tensors"]:
        if t["ggml_type_id"] == 14:
            raw = container.tensor_bytes(blob, t)
            one = quants.dequant_q6_k(raw[:210])
            ok, _ = ref.close(one, truth[t["name"]].reshape(-1)[:256])
            out["q6k_block_match"] = 1.0 if ok else 0.0
            got = quants.dequant_tensor(raw, 14, t["n_elements"])
            ok, err = ref.close(got, truth[t["name"]])
            out["q6k_match"] = 1.0 if ok else 0.0
            out["q6k_rel_err"] = err
        elif t["ggml_type_id"] == 0:
            raw = container.tensor_bytes(blob, t)
            got = quants.dequant_tensor(raw, 0, t["n_elements"])
            ok, _ = ref.close(got, truth[t["name"]], tol=0.0)
            out["f32_match"] = 1.0 if ok else 0.0
    return out
