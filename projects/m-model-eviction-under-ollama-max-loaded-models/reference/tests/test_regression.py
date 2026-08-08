import sys

sys.path.insert(0, ".")
from ollama_evict.tracker import track_model
from ollama_evict.policy import select_eviction
from ollama_evict.engine import process_request


def test_eviction_targets_oldest_timestamp():
    state = {}
    track_model(state, "model_a", 1000, 10)
    track_model(state, "model_b", 1000, 20)
    track_model(state, "model_c", 1000, 30)
    victim = select_eviction(state, 2)
    assert victim == "model_a", f"expected model_a to be evicted, got {victim}"


def test_capacity_never_exceeded_after_engine_run():
    state = {}
    process_request(state, "model_a", 100, 2, 1)
    process_request(state, "model_b", 100, 2, 2)
    process_request(state, "model_c", 100, 2, 3)
    loaded = [m for m, d in state.items() if d.get("loaded", True)]
    assert len(loaded) <= 2, f"too many models loaded: {loaded}"


def test_track_updates_timestamp_correctly():
    state = {}
    track_model(state, "model_a", 500, 10)
    track_model(state, "model_a", 500, 40)
    assert state["model_a"]["last_used"] == 40
    assert state["model_a"]["access_count"] == 2
