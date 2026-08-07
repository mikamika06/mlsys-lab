import ref

def check(workdir):
    from zerothree.schedule import simulate_all_gather_free_cycle

    out = {"schedule_valid": 0.0, "peak_memory_bounded": 0.0}

    layer_sizes = ref.LAYER_CONFIGS[0]
    dp_degree = 8

    try:
        want = ref.simulate_all_gather_free_cycle(layer_sizes, dp_degree)
        got = simulate_all_gather_free_cycle(layer_sizes, dp_degree)
    except Exception as e:
        out["_note"] = f"schedule function raised {type(e).__name__}: {str(e)[:100]}"
        return out

    if not isinstance(got, dict) or "peak_memory" not in got or "timeline" not in got:
        out["_note"] = "return value must be a dictionary with 'peak_memory' and 'timeline'"
        return out

    if abs(got["peak_memory"] - want["peak_memory"]) < 1e-2:
        out["peak_memory_bounded"] = 1.0
    else:
        out["_note"] = f"peak_memory got {got['peak_memory']}, reference {want['peak_memory']}"
        return out

    if len(got["timeline"]) == len(want["timeline"]):
        out["schedule_valid"] = 1.0
    else:
        out["_note"] = f"timeline length mismatch: got {len(got['timeline'])}, reference {len(want['timeline'])}"

    return out
