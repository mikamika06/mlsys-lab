import sys
sys.path.insert(0, ".")
from kvsim.simulator import simulate_pipeline
from kvsim.validator import validate_kv_transfer_config
from kvsim.faults import analyze_nixl_faults


def test_simulation_completes_all_requests():
    reqs = [{"id": "1", "prompt_tokens": 50, "kv_size_bytes": 1000, "decode_tokens": 10}]
    res = simulate_pipeline(reqs, 1, 1, 100000000)
    assert len(res) == 1
    assert "completion_time" in res[0]


def test_validator_rejects_duplicates():
    cfg = {
        "kv_connector": "NixlConnector",
        "roles": [
            {"rank": 0, "role": "prefill"},
            {"rank": 0, "role": "decode"}
        ]
    }
    assert validate_kv_transfer_config(cfg)["valid"] is False


def test_fault_analysis_metadata_crash():
    topo = {"nodes": [{"id": "n1", "role": "prefill", "tier": "core"}]}
    res = analyze_nixl_faults(topo, "server_crash")
    assert res["blast_radius"] == "cluster_wide"
    assert len(res["affected_nodes"]) > 0
