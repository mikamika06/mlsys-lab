import ref


def check(workdir):
    from pipeline.stages import StageOne, StageTwo, PipelineOrchestrator

    out = {"pipeline_matched": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        s1 = StageOne()
        s2 = StageTwo()
        orch = PipelineOrchestrator(s1, s2)
        res = orch.run(cfg["input"])
        if isinstance(res, dict) and res.get("result") == cfg["expected"]:
            ok += 1
    out["pipeline_matched"] = float(ok)
    return out
