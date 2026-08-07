import ref
from mlxlora.eval_pitfall import measure_execution

def check(workdir):
    out = {"pitfall_reproduced": 0.0, "lazy_eval_correct": 0.0}

    root, expected_val = ref.generate_sample_graph()

    unforced_res = measure_execution(root, force_eval=False)
    ref_unforced = ref.measure_execution(root, force_eval=False)

    root2, _ = ref.generate_sample_graph()
    forced_res = measure_execution(root2, force_eval=True)
    ref_forced = ref.measure_execution(root2, force_eval=True)

    if (isinstance(unforced_res, dict) and
        unforced_res.get("computed") is False and
        unforced_res.get("evaluated_nodes", 0) < unforced_res.get("total_nodes", 0) and
        unforced_res.get("total_nodes") == ref_unforced["total_nodes"]):
        out["pitfall_reproduced"] = 1.0

    if (isinstance(forced_res, dict) and
        forced_res.get("computed") is True and
        forced_res.get("evaluated_nodes") == forced_res.get("total_nodes") and
        forced_res.get("total_nodes") == ref_forced["total_nodes"] and
        forced_res.get("result") is not None and
        ref.np.allclose(forced_res["result"], expected_val)):
        out["lazy_eval_correct"] = 1.0

    if "_note" not in out and (out["pitfall_reproduced"] == 0.0 or out["lazy_eval_correct"] == 0.0):
        out["_note"] = f"unforced: {unforced_res}, forced: {forced_res}"

    return out
