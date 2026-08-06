from onednn_diag.fallback import analyze_fallback_causes


def test_detect_masked_fallback():
    sample_logs = [
        "onednn,exec,cpu,jit:avx512,forward,src_f32...exec_time:1.2",
        "onednn,exec,cpu,ref:any,forward,src_bf16...exec_time:15.4",
        "onednn,exec,cpu,jit:amx_bf16,forward,src_bf16...exec_time:0.8"
    ]
    results = analyze_fallback_causes(sample_logs)
    assert len(results) >= 1
    assert any(r["implementation"].startswith("ref") for r in results)
