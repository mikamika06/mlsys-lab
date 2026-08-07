def max_valid_tp(config: dict, max_tp: int = 64) -> int:
    attn_heads = config["num_attention_heads"]
    kv_heads = config["num_kv_heads"]
    hidden = config["hidden_size"]
    inter = config["intermediate_size"]

    best_tp = 1
    tp = 1
    while tp <= max_tp:
        if (
            attn_heads % tp == 0
            and kv_heads % tp == 0
            and hidden % tp == 0
            and inter % tp == 0
        ):
            best_tp = tp
        tp *= 2
    return best_tp


def get_valid_tp_degrees(config: dict, max_tp: int = 64) -> list[int]:
    attn_heads = config["num_attention_heads"]
    kv_heads = config["num_kv_heads"]
    hidden = config["hidden_size"]
    inter = config["intermediate_size"]

    valid = []
    tp = 1
    while tp <= max_tp:
        if (
            attn_heads % tp == 0
            and kv_heads % tp == 0
            and hidden % tp == 0
            and inter % tp == 0
        ):
            valid.append(tp)
        tp *= 2
    return valid
