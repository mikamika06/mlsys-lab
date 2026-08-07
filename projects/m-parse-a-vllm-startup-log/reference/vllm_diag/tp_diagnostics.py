from vllm_diag.tp_checker import check_tp_sharding


def diagnose_tp_garbage_output(model_config, tp_size, log_data):
    """Diagnose potential causes of garbage output at TP > 1."""
    if tp_size <= 1:
        return {"issue": None, "details": "No TP issue on single GPU"}

    sharding_check = check_tp_sharding(model_config, tp_size)
    if not sharding_check["valid"]:
        return {"issue": "INVALID_SHARDING", "details": sharding_check["reason"]}

    vocab_size = model_config.get("vocab_size", 0)
    pad_vocab_to_multiple = model_config.get("pad_vocab_to_multiple_of", 1)

    if vocab_size % tp_size != 0 and (pad_vocab_to_multiple % tp_size != 0):
        return {
            "issue": "UNPADDED_VOCAB_MISMATCH",
            "details": f"vocab_size {vocab_size} is not divisible by tp_size {tp_size} and padding multiple {pad_vocab_to_multiple} does not align"
        }

    num_heads = model_config.get("num_attention_heads", 0)
    num_kv_heads = model_config.get("num_key_value_heads", num_heads)

    if num_kv_heads < tp_size and (tp_size % num_kv_heads == 0):
        if not model_config.get("enable_kv_head_repeat", False):
            return {
                "issue": "KV_HEAD_REPETITION_MISSING",
                "details": f"num_kv_heads ({num_kv_heads}) < tp_size ({tp_size}) requires explicit KV head repetition"
            }

    warnings = log_data.get("warnings", [])
    for w in warnings:
        if "nccl" in w.lower() or "communication" in w.lower():
            return {"issue": "NCCL_DESYNC_WARNING", "details": w}

    return {"issue": None, "details": "No known TP garbage output issue detected"}
