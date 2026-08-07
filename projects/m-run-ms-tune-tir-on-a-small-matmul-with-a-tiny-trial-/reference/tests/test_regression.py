import sys
import os

sys.path.insert(0, ".")
from tuntir.tune import rank_candidates


def test_ranking_order():
    dummy_data = [
        {"id": 0, "latency": 0.050},
        {"id": 1, "latency": 0.010},
        {"id": 2, "latency": 0.030}
    ]
    import tempfile
    import json
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "db.json")
        with open(path, "w") as f:
            json.dump(dummy_data, f)
        res = rank_candidates(path)
        latencies = [x["latency"] for x in res]
        assert latencies == sorted(latencies), f"candidates not sorted by latency: {latencies}"
        assert len(res) <= 5
