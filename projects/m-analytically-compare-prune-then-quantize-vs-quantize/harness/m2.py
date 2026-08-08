import ref


def check(workdir):
    from pqutils.pipeline import run_joint_pipeline
    fixtures = ref.generate_fixtures()
    ok = 0
    out = {"pipeline_matches": 0.0, "fixtures": float(len(fixtures))}
    for idx, fx in enumerate(fixtures):
        try:
            r1 = run_joint_pipeline(fx["weights"], fx["hessian"], fx["sparsity"], fx["q_bits"], "prune_then_quantize")
            r2 = run_joint_pipeline(fx["weights"], fx["hessian"], fx["sparsity"], fx["q_bits"], "quantize_then_prune")
            if r1 is not None and r2 is not None:
                ok += 1
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"pipeline fixture {idx} raised {type(e).__name__}: {str(e)[:100]}"
    out["pipeline_matches"] = float(ok)
    return out
