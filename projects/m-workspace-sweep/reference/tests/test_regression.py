import sys

sys.path.insert(0, ".")
from sweep.engine import plan_engine, sweep_workspace

def test_rejects_tactic_too_large_at_max_shape():
    profile = {"max_s": 10, "opt_s": 1}
    config = {
        "weights_memory": 0,
        "layers": [{
            "tactics": [
                {"base_ws": 0, "ws_factor": 100, "base_lat": 10, "lat_factor": 0}
            ]
        }]
    }

    mem, lat = plan_engine(config, profile, 500)

    assert mem == float('inf'), "Engine should reject tactics that exceed workspace limit at max_s"

def test_sweep_returns_first_on_tie():
    profile = {"max_s": 10, "opt_s": 10}
    config = {
        "weights_memory": 0,
        "layers": [{
            "tactics": [
                {"base_ws": 10, "ws_factor": 1, "base_lat": 10, "lat_factor": 0}
            ]
        }]
    }

    idx = sweep_workspace(config, profile, 1000, [500, 1000])

    assert idx == 0, "Sweep should prefer the lower index (tighter limit) on a tie"
