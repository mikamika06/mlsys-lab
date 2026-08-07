from disagg.p1d import DecodeWorker, Pipeline1P1D, PrefillWorker
from disagg.verify import analyze_execution_metrics, verify_decode_skips_prefill

TEST_REQUESTS = [
    {"id": "req_alpha", "prompt": list(range(100)), "steps": 20},
    {"id": "req_beta", "prompt": list(range(256)), "steps": 15},
    {"id": "req_gamma", "prompt": list(range(512)), "steps": 10}
]


def run_oracle_pipeline(num_layers=12, head_dim=64, num_heads=12):
    pw = PrefillWorker(0, num_layers, head_dim, num_heads)
    dw = DecodeWorker(1, num_layers, head_dim, num_heads)
    pipe = Pipeline1P1D(pw, dw)

    results = []
    for req in TEST_REQUESTS:
        res = pipe.process_request(req["id"], req["prompt"], req["steps"])
        results.append(res)
    return results
