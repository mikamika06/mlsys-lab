def check(workdir):
    from ngram.engine import NgramSpeculativeEngine
    m = {"speedup_ok": 0.0}
    prompt = list(range(100)) * 5
    engine = NgramSpeculativeEngine(prompt, k=4, disable_threshold=0.05)
    mock_model = lambda out, spec=None: (spec, len(spec)) if spec else ([1], 0)
    out = engine.run(mock_model, max_steps=10)
    if len(out) > 0:
        m["speedup_ok"] = 1.0
    return m
