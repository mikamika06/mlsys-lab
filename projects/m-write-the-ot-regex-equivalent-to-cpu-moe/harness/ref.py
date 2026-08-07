CONFIGS = [
    {"flag": "--cpu-moe", "expected": "^blk\\.\\d+\\.ffn_(gate|up|down)_expts\\..*"},
    {"flag": "--cpu-moe --layer 5", "expected": "^blk\\.5\\.ffn_(gate|up|down)_expts\\..*"},
    {"flag": "--cpu-moe --expert-prefix", "expected": "^blk\\.\\d+\\.ffn_.*expts.*"},
]

def translate_flag(flag: str) -> str:
    if "--layer" in flag:
        parts = flag.split()
        idx = parts.index("--layer")
        layer_num = parts[idx + 1]
        return f"^blk\\.{layer_num}\\.ffn_(gate|up|down)_expts\\..*"
    if "--expert-prefix" in flag:
        return "^blk\\.\\d+\\.ffn_.*expts.*"
    return "^blk\\.\\d+\\.ffn_(gate|up|down)_expts\\..*"

def check_specificity(regex: str) -> bool:
    import re
    test_tensor_moe = "blk.0.ffn_gate_expts.weight"
    test_tensor_dense = "blk.0.attn_q.weight"
    return bool(re.match(regex, test_tensor_moe)) and not bool(re.match(regex, test_tensor_dense))
