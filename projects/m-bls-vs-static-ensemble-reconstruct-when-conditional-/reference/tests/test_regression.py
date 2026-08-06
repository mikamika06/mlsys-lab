import numpy as np
from bls_router.router import route_request

def test_bls_vs_static_routing():
    req_high = {"id": 1, "data": np.array([10.0, 10.0], dtype=np.float32)}
    res_high = route_request(req_high)
    assert res_high["branch"] == "model_a"
    assert np.allclose(res_high["result"], np.array([20.0, 20.0], dtype=np.float32))
    assert "extra_branch_executed" not in res_high

    req_low = {"id": 2, "data": np.array([1.0, 1.0], dtype=np.float32)}
    res_low = route_request(req_low)
    assert res_low["branch"] == "model_b"
    assert np.allclose(res_low["result"], np.array([11.0, 11.0], dtype=np.float32))
    assert "extra_branch_executed" not in res_low
