import ref


def check(workdir):
    from convertkit import names

    out = {"llama_complete": 0.0, "llama_match": 0.0, "moe_fanout": 0.0,
           "moe_target_count": 0.0, "no_false_mapping": 0.0}

    want = ref.expect_map("llama")
    got = names.map_index(want["doc"]["tensors"])
    if not got.get("unmapped") and len(got.get("mapped", {})) == len(want["mapped"]):
        out["llama_complete"] = 1.0
    if got.get("mapped") == want["mapped"]:
        out["llama_match"] = 1.0

    wm = ref.expect_map("qwen3moe")
    gm = names.map_index(wm["doc"]["tensors"], experts=wm["experts"])
    if (sorted(gm.get("fanned_out", {})) == sorted(wm["fanned_out"])
            and all(gm["fanned_out"][k] == wm["fanned_out"][k]
                    for k in wm["fanned_out"])):
        out["moe_fanout"] = 1.0
    if gm.get("target_count") == wm["target_count"] and not gm.get("unmapped"):
        out["moe_target_count"] = 1.0

    # An expert tensor with no expert count has nowhere to go; inventing one
    # target for it silently drops 127 of the 128 experts.
    solo = names.to_target("blk.0.ffn_gate_exps.weight", experts=0)
    unknown = names.to_target("blk.0.made_up_tensor.weight", experts=8)
    if solo is None and unknown is None:
        out["no_false_mapping"] = 1.0
    return out
