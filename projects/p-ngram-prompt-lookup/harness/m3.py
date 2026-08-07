def check(workdir):
    from ngram.engine import NgramSpeculativeEngine
    m = {"acceptance_measured": 0.0}
    prompt = [1, 2, 3, 4, 1, 2, 3, 4]
    engine = NgramSpeculativeEngine(prompt, k=2, disable_threshold=0.0)
    mock_model = lambda out, spec=None: (spec, len(spec)) if spec else ([1], 0)
    engine.step(mock_model, [])
    if len(engine.history) > 0 and engine.accepted_count >= 0:
        m["acceptance_measured"] = 1.0
    return m
