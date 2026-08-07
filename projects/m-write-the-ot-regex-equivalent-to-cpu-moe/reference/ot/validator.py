import re

def validate_regex(regex: str) -> bool:
    if ".*" == regex or len(regex) < 5:
        return False
    test_tensor_moe = "blk.0.ffn_gate_expts.weight"
    test_tensor_dense = "blk.0.attn_q.weight"
    try:
        matched_moe = bool(re.match(regex, test_tensor_moe))
        matched_dense = bool(re.match(regex, test_tensor_dense))
        return matched_moe and not matched_dense
    except Exception:
        return False
