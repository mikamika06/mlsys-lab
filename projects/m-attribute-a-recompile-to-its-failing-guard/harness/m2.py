import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import harness.ref as ref


def check(workdir):
    from guardeval.attribution import Engine

    out = {"recompiles_matched": 0.0, "attributions_matched": 0.0}

    stream, compile_fn = ref.generate_test_cases()

    ref_engine = Engine(compile_fn)
    ref_res = ref_engine.process_stream(stream)

    learner_engine = Engine(compile_fn)
    learner_res = learner_engine.process_stream(stream)

    if learner_res["recompile_count"] == ref_res["recompile_count"]:
        out["recompiles_matched"] = 1.0
    else:
        out["_note"] = f"Recompile count mismatch: got {learner_res['recompile_count']}, want {ref_res['recompile_count']}"

    ref_reasons = [a["reason"] for a in ref_res["attributions"]]
    learner_reasons = [a["reason"] for a in learner_res.get("attributions", [])]

    if ref_reasons == learner_reasons:
        out["attributions_matched"] = 1.0
    elif out["recompiles_matched"] == 1.0:
        out["_note"] = f"Attribution reasons mismatch: got {learner_reasons[:3]}, want {ref_reasons[:3]}"

    return out
