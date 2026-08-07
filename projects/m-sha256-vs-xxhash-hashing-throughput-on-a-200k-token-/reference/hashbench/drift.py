def simulate_template_drift(base_prompt: str, drift_type: str) -> str:
    if drift_type == "whitespace":
        return base_prompt.replace(" ", "  ")
    elif drift_type == "system_prompt":
        return "<|system|>\nYou are a helpful assistant.\n" + base_prompt
    elif drift_type == "none":
        return base_prompt
    else:
        return base_prompt + "\n"
