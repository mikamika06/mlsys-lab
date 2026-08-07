def translate_cpu_moe(flag: str) -> str:
    if "--layer" in flag:
        parts = flag.split()
        idx = parts.index("--layer")
        layer_num = parts[idx + 1]
        return f"^blk\\.{layer_num}\\.ffn_(gate|up|down)_expts\\..*"
    if "--expert-prefix" in flag:
        return "^blk\\.\\d+\\.ffn_.*expts.*"
    return "^blk\\.\\d+\\.ffn_(gate|up|down)_expts\\..*"
