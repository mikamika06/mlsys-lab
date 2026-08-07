def test_pipeline_validity():
    from app.pipeline import run_1000_times
    results = run_1000_times()
    for tokens in results:
        assert len(tokens) == 9
        assert tokens[0] == 0
        assert tokens[-1] == 1
