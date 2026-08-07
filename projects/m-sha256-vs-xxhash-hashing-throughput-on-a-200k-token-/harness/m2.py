import ref


def check(workdir):
    from hashbench.drift import simulate_template_drift

    prompt = "The quick brown fox jumps over the lazy dog."
    drift_types = ["none", "whitespace", "system_prompt"]
    matches = 0
    for dt in drift_types:
        want = ref.simulate_template_drift(prompt, dt)
        got = simulate_template_drift(prompt, dt)
        if want == got:
            matches += 1

    matched = 1.0 if matches == len(drift_types) else 0.0
    return {
        "drift_penalty_matched": matched,
        "_note": f"Matched {matches}/{len(drift_types)} drift simulations"
    }
