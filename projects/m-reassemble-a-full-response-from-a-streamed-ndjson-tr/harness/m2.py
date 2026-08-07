import ref

def check(workdir):
    from rawstream.delta import quantify_delta
    prompts = [
        ("print(1)", "You are a coding assistant."),
        ("hello world", ""),
        ("explain quantum physics", "Be concise.")
    ]
    matched = True
    for p, sp in prompts:
        want = ref.compute_delta(p, sp)
        try:
            got = quantify_delta(p, system_prompt=sp)
        except Exception:
            matched = False
            break
        if not isinstance(got, dict) or got.get("char_delta") != want["char_delta"]:
            matched = False
            break
    return {"delta_matched": 1.0 if matched else 0.0}
