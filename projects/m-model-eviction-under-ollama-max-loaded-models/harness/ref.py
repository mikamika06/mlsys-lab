SCENARIOS = [
    {
        "ops": [
            ("model_a", 1000, 10),
            ("model_b", 2000, 20),
            ("model_c", 1500, 30),
        ],
        "max_loaded": 2,
    },
    {
        "ops": [
            ("llama3", 4000, 100),
            ("mistral", 3000, 110),
            ("phi3", 2000, 120),
            ("gemma", 2500, 130),
        ],
        "max_loaded": 2,
    },
    {
        "ops": [
            ("m1", 500, 1),
            ("m1", 500, 5),
            ("m2", 500, 10),
            ("m3", 500, 15),
        ],
        "max_loaded": 2,
    }
]

def run_oracle(scenario):
    state = {}
    history = []
    for name, mem, ts in scenario["ops"]:
        if name in state:
            state[name]["last_used"] = ts
            state[name]["access_count"] += 1
        else:
            state[name] = {"memory_bytes": mem, "last_used": ts, "access_count": 1, "loaded": True}

        while len([m for m, d in state.items() if d.get("loaded", True)]) > scenario["max_loaded"]:
            loaded = [m for m, d in state.items() if d.get("loaded", True)]
            victim = min(loaded, key=lambda x: (state[x]["last_used"], state[x]["access_count"]))
            state[victim]["loaded"] = False
            history.append(victim)
    return state
