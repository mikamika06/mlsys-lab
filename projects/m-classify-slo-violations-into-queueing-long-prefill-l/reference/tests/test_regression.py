from slo.classifier import classify_violations


def test_classification_logic():
    reqs = [{
        "id": "r1",
        "arrival_time": 0.0,
        "start_time": 0.1,
        "prefill_end_time": 0.2,
        "finish_time": 2.0
    }]
    res = classify_violations(reqs, 1.0)
    assert res.get("r1") == "long_output"
