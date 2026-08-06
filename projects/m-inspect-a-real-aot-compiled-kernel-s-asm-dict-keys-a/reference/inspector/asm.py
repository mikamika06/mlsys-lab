def analyze_asm_dict(asm: dict) -> dict:
    return {k: len(v) for k, v in asm.items()}
