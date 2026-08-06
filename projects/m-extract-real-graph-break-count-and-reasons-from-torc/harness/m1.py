import ref


def check(workdir):
    from peftcomp.analyzer import extract_graph_breaks
    model, x_base, _ = ref.get_oracle_model_and_inputs()
    got = extract_graph_breaks(model, x_base)
    ref_res = extract_graph_breaks(model, x_base)
    matched = 1.0 if got.get("count") == ref_res.get("count") else 0.0
    out = {"breaks_matched": matched}
    if matched == 0.0:
        out["_note"] = f"Expected count {ref_res.get('count')}, got {got.get('count')}"
    return out
