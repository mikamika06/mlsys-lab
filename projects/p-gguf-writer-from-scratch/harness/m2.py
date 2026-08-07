from gguf_writer.tensor_map import map_tensor_name

def check(workdir):
    m = {"tensor_mapping_ok": 0.0}
    try:
        res1 = map_tensor_name("model.embed_tokens.weight")
        res2 = map_tensor_name("lm_head.weight")
        if res1 == "token_embd.weight" and res2 == "output.weight":
            m["tensor_mapping_ok"] = 1.0
    except Exception:
        pass
    return m
