def identify_flaws(script_text):
    flaws = []
    if "warmup" not in script_text or "time.sleep" in script_text:
        flaws.append({"flaw": "cold_start_included", "bias": "positive"})
    if "print(" in script_text or "sys.stdout.write" in script_text:
        flaws.append({"flaw": "synchronous_logging", "bias": "negative"})
    if "queue" in script_text:
        flaws.append({"flaw": "unbounded_queue", "bias": "positive"})
    return sorted(flaws, key=lambda x: x["flaw"])


def cold_start_inflation(cold_elapsed, steady_elapsed, total_tokens):
    cold_toks = total_tokens / cold_elapsed
    steady_toks = total_tokens / steady_elapsed
    return ((cold_toks - steady_toks) / steady_toks) * 100.0
