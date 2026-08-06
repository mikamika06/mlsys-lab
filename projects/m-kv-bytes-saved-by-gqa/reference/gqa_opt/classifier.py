def classify_attention(config: dict) -> dict:
    q_heads = config["num_attention_heads"]
    kv_heads = config.get("num_key_value_heads", q_heads)

    if kv_heads == q_heads:
        family = "MHA"
    elif kv_heads == 1:
        family = "MQA"
    elif 1 < kv_heads < q_heads:
        family = "GQA"
    else:
        raise ValueError("Invalid head configuration")

    ratio = q_heads // kv_heads
    return {
        "family": family,
        "num_q_heads": q_heads,
        "num_kv_heads": kv_heads,
        "group_ratio": ratio,
    }


def compute_group_mapping(config: dict) -> list[int]:
    q_heads = config["num_attention_heads"]
    kv_heads = config.get("num_key_value_heads", q_heads)
    ratio = q_heads // kv_heads
    mapping = []
    for q_idx in range(q_heads):
        mapping.append(q_idx // ratio)
    return mapping
