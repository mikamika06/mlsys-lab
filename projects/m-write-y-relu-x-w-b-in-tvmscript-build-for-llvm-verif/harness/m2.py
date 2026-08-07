import ref


def check(workdir):
    from tvm_pipeline.importer import import_torch_mlp
    from tvm_pipeline.inspect import count_relax_ops

    out = {
        "imported_matches": 0.0,
        "call_tir_matched": 0.0,
        "raw_ops_matched": 0.0
    }

    ok_imp = 0
    ok_tir = 0
    ok_raw = 0
    total = len(ref.EXPORTED_MODELS)

    for i, model in enumerate(ref.EXPORTED_MODELS):
        try:
            res = import_torch_mlp(model)
            if res is not None:
                ok_imp += 1
            counts = count_relax_ops(res if res is not None else model)
            expected_tir = sum(1 for n in model.graph_nodes if n == "call_tir")
            expected_raw = sum(1 for n in model.graph_nodes if n == "raw_op")

            if counts.get("call_tir") == expected_tir:
                ok_tir += 1
            if counts.get("raw_ops") == expected_raw:
                ok_raw += 1
        except Exception as e:
            out["_note"] = f"Error evaluating model {i}: {e}"
            return out

    out["imported_matches"] = 1.0 if ok_imp == total else 0.0
    out["call_tir_matched"] = 1.0 if ok_tir == total else 0.0
    out["raw_ops_matched"] = 1.0 if ok_raw == total else 0.0
    return out
