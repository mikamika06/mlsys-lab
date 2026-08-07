import re


def check_feasibility(file_list: list[str]) -> dict:
    if "config.json" not in file_list:
        return {"feasible": False, "reason": "missing_config"}

    tok_found = any(
        f in file_list
        for f in ("tokenizer.json", "tokenizer.model", "vocab.json")
    )
    if not tok_found:
        return {"feasible": False, "reason": "missing_tokenizer"}

    st_shards = []
    pt_shards = []
    for f in file_list:
        m_st = re.match(r"^(?:model)-(\d{5})-of-(\d{5})\.safetensors$", f)
        if m_st:
            st_shards.append((int(m_st.group(1)), int(m_st.group(2))))
        m_pt = re.match(r"^(?:pytorch_model)-(\d{5})-of-(\d{5})\.bin$", f)
        if m_pt:
            pt_shards.append((int(m_pt.group(1)), int(m_pt.group(2))))

    if st_shards:
        total = st_shards[0][1]
        found = set(s[0] for s in st_shards)
        missing = [i for i in range(1, total + 1) if i not in found]
        if missing or len(st_shards) != total:
            return {
                "feasible": False,
                "reason": "missing_shards",
                "missing_indices": missing,
            }
        return {
            "feasible": True,
            "reason": "ok",
            "format": "safetensors",
            "shard_count": total,
        }

    if pt_shards:
        total = pt_shards[0][1]
        found = set(s[0] for s in pt_shards)
        missing = [i for i in range(1, total + 1) if i not in found]
        if missing or len(pt_shards) != total:
            return {
                "feasible": False,
                "reason": "missing_shards",
                "missing_indices": missing,
            }
        return {
            "feasible": True,
            "reason": "ok",
            "format": "pytorch",
            "shard_count": total,
        }

    if "model.safetensors" in file_list:
        return {
            "feasible": True,
            "reason": "ok",
            "format": "safetensors",
            "shard_count": 1,
        }

    if "pytorch_model.bin" in file_list:
        return {
            "feasible": True,
            "reason": "ok",
            "format": "pytorch",
            "shard_count": 1,
        }

    return {"feasible": False, "reason": "missing_weights"}
