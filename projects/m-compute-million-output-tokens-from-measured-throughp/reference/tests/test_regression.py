from capacity.sizing import compute_required_gpus

def test_sizing_headroom_invariant():
    """Test required capacity safety margins."""
    hourly_traffic = [1000.0, 5000.0, 10000.0, 8000.0]
    throughput = 2000.0
    gpus_per_inst = 4
    res = compute_required_gpus(hourly_traffic, throughput, gpus_per_inst, headroom_fraction=0.30)
    assert res["target_capacity_tok_per_sec"] >= max(hourly_traffic) * 1.30
    assert res["instances_needed"] * throughput >= res["target_capacity_tok_per_sec"]
    assert res["total_gpus"] == res["instances_needed"] * gpus_per_inst
