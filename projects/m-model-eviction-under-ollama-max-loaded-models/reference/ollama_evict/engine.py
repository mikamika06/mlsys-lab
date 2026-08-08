from ollama_evict.tracker import track_model
from ollama_evict.policy import select_eviction


def process_request(state, model_name, memory_bytes, max_loaded, timestamp):
    track_model(state, model_name, memory_bytes, timestamp)
    evicted = []
    while len([m for m, d in state.items() if d.get("loaded", True)]) > max_loaded:
        victim = select_eviction(state, max_loaded)
        if victim:
            state[victim]["loaded"] = False
            evicted.append(victim)
        else:
            break
    return evicted
