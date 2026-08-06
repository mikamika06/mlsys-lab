def generate_repro(config: dict, error_code: str) -> str:
    lines = [
        "import torch",
        "import flash_attn",
        f"# Error: {error_code}",
        f"config = {config!r}",
        "print('Running minimal reproduction...')",
    ]
    return "\n".join(lines)
