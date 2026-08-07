import ref


def check(workdir):
    from gguf_interop.modelfile import generate_modelfile
    from gguf_interop.size import predict_quant_file_size

    out = {"modelfile_match": 0.0, "size_match": 0.0}

    path = ref.MODELFILE_PARAMS["gguf_path"]
    params = ref.MODELFILE_PARAMS["params"]
    got_mf = generate_modelfile(path, params)
    want_mf = ref.oracle_generate_modelfile(path, params)

    if got_mf == want_mf:
        out["modelfile_match"] = 1.0
    else:
        out["_note"] = f"modelfile mismatch:\ngot:\n{got_mf}\nwant:\n{want_mf}"

    t_infos = ref.TENSOR_LIST
    got_size = predict_quant_file_size(t_infos, "Q4_K_M")
    want_size = ref.oracle_predict_quant_file_size(t_infos, "Q4_K_M")

    if got_size == want_size:
        out["size_match"] = 1.0
    elif "_note" not in out:
        out["_note"] = f"size mismatch: got {got_size}, want {want_size}"

    return out
