from pipeline.stages import StageOne, StageTwo, PipelineOrchestrator


def test_pipeline_regression():
    s1 = StageOne()
    s2 = StageTwo()
    orch = PipelineOrchestrator(s1, s2)
    res = orch.run([1, 2, 3])
    assert "result" in res
    assert res["result"] == [3, 5, 7]
    meta = res["meta"]
    assert "stage_one_start" in meta
    assert "stage_one_end" in meta
    assert "stage_two_start" in meta
    assert "stage_two_end" in meta
    assert meta["stage_one_end"] >= meta["stage_one_start"]
    assert meta["stage_two_end"] >= meta["stage_two_start"]
