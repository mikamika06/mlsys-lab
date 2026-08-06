import ref


def check(workdir):
    from pipeline.stages import StageOne, StageTwo, PipelineOrchestrator

    out = {"latency_tracked": 0.0, "stage_isolated": 0.0}
    s1 = StageOne()
    s2 = StageTwo()
    orch = PipelineOrchestrator(s1, s2)
    res = orch.run(ref.CONFIGS[0]["input"])
    meta = res.get("meta", {})

    has_latencies = all(k in meta for k in ["stage_one_start", "stage_one_end", "stage_two_start", "stage_two_end"])
    if has_latencies:
        out["latency_tracked"] = 1.0

    s1_dur = meta.get("stage_one_end", 0) - meta.get("stage_one_start", 0)
    s2_dur = meta.get("stage_two_end", 0) - meta.get("stage_two_start", 0)
    if s1_dur >= 0 and s2_dur >= 0:
        out["stage_isolated"] = 1.0

    return out
