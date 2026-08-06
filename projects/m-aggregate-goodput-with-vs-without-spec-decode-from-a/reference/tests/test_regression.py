def test_goodput_rejection_penalty():
    from specdiag.goodput import compute_request_goodput

    req_with_rejected = {
        "duration_ms": 1000,
        "latency_ms": 100,
        "sla_latency_ms": 200,
        "accepted_tokens": 100,
        "rejected_tokens": 40,
    }
    res_penalized = compute_request_goodput(req_with_rejected, penalty_factor=0.5)
    res_zero_pen = compute_request_goodput(req_with_rejected, penalty_factor=0.0)

    assert res_penalized["goodput_tps"] == 80.0
    assert res_zero_pen["goodput_tps"] == 100.0
    assert res_penalized["goodput_tps"] < res_zero_pen["goodput_tps"]


def test_goodput_sla_cutoff():
    from specdiag.goodput import compute_request_goodput

    req_violates_sla = {
        "duration_ms": 1000,
        "latency_ms": 350,
        "sla_latency_ms": 200,
        "accepted_tokens": 100,
        "rejected_tokens": 0,
    }
    res = compute_request_goodput(req_violates_sla, penalty_factor=0.5)
    assert res["goodput_tps"] == 0.0
    assert not res["meets_sla"]
