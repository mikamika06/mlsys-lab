import math
import re
import zlib


def check_feasibility(file_list):
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


def compute_chkhsh(tokens, pre_tokenizer):
    data = pre_tokenizer.encode("utf-8") + b"\x00"
    for tok in tokens:
        tb = tok.encode("utf-8")
        data += len(tb).to_bytes(4, "little") + tb
    return zlib.crc32(data) & 0xFFFFFFFF


def _dtype_bytes(dt):
    if dt in ("float32", "f32"):
        return 4
    if dt in ("float16", "f16", "bfloat16", "bf16"):
        return 2
    if dt in ("int8", "i8", "q8_0"):
        return 1
    return 4


def estimate_conversion_memory(tensors, lazy=True, base_overhead_mb=256.0):
    base_bytes = int(base_overhead_mb * 1024 * 1024)
    if not tensors:
        return {
            "peak_memory_bytes": base_bytes,
            "lazy": bool(lazy),
            "total_model_bytes": 0,
        }

    sizes = []
    shard_map = {}
    for t in tensors:
        sz = math.prod(t["shape"]) * _dtype_bytes(t.get("dtype", "float32"))
        sizes.append(sz)
        sid = t.get("shard_id", "default")
        if sid is None:
            sid = "default"
        shard_map.setdefault(sid, []).append(sz)

    total_bytes = sum(sizes)

    if not lazy:
        max_t = max(sizes)
        peak = base_bytes + total_bytes + max_t
    else:
        shard_peaks = []
        for sid, s_sizes in shard_map.items():
            shard_peaks.append(sum(s_sizes) + max(s_sizes))
        peak = base_bytes + max(shard_peaks)

    return {
        "peak_memory_bytes": int(peak),
        "lazy": bool(lazy),
        "total_model_bytes": int(total_bytes),
    }


FILES_TEST_CASES = [
    (
        [
            "config.json",
            "tokenizer.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ],
        check_feasibility(
            [
                "config.json",
                "tokenizer.json",
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ]
        ),
    ),
    (
        ["config.json", "tokenizer.model", "model.safetensors"],
        check_feasibility(
            ["config.json", "tokenizer.model", "model.safetensors"]
        ),
    ),
    (
        [
            "config.json",
            "vocab.json",
            "model-00001-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        ],
        check_feasibility(
            [
                "config.json",
                "vocab.json",
                "model-00001-of-00003.safetensors",
                "model-00003-of-00003.safetensors",
            ]
        ),
    ),
    (
        ["tokenizer.json", "model.safetensors"],
        check_feasibility(["tokenizer.json", "model.safetensors"]),
    ),
    (
        ["config.json", "model.safetensors"],
        check_feasibility(["config.json", "model.safetensors"]),
    ),
    (
        ["config.json", "tokenizer.json"],
        check_feasibility(["config.json", "tokenizer.json"]),
    ),
    (
        [
            "config.json",
            "tokenizer.json",
            "pytorch_model-00001-of-00002.bin",
            "pytorch_model-00002-of-00002.bin",
        ],
        check_feasibility(
            [
                "config.json",
                "tokenizer.json",
                "pytorch_model-00001-of-00002.bin",
                "pytorch_model-00002-of-00002.bin",
            ]
        ),
    ),
]

TOKENIZER_TEST_CASES = [
    (
        ["[PAD]", "[UNK]", "hello", "world", "gguf"],
        "llama-bpe",
        compute_chkhsh(["[PAD]", "[UNK]", "hello", "world", "gguf"], "llama-bpe"),
    ),
    (
        ["<|endoftext|>", "qwen", "tokenization"],
        "qwen2",
        compute_chkhsh(["<|endoftext|>", "qwen", "tokenization"], "qwen2"),
    ),
]

MEMORY_TEST_CASES = [
    (
        [
            {"name": "t1", "shape": [2048, 2048], "dtype": "float32", "shard_id": 1},
            {"name": "t2", "shape": [2048, 2048], "dtype": "float32", "shard_id": 1},
            {"name": "t3", "shape": [2048, 2048], "dtype": "float32", "shard_id": 2},
            {"name": "t4", "shape": [2048, 2048], "dtype": "float32", "shard_id": 2},
        ],
        estimate_conversion_memory(
            [
                {"name": "t1", "shape": [2048, 2048], "dtype": "float32", "shard_id": 1},
                {"name": "t2", "shape": [2048, 2048], "dtype": "float32", "shard_id": 1},
                {"name": "t3", "shape": [2048, 2048], "dtype": "float32", "shard_id": 2},
                {"name": "t4", "shape": [2048, 2048], "dtype": "float32", "shard_id": 2},
            ],
            lazy=True,
        ),
        estimate_conversion_memory(
            [
                {"name": "t1", "shape": [2048, 2048], "dtype": "float32", "shard_id": 1},
                {"name": "t2", "shape": [2048, 2048], "dtype": "float32", "shard_id": 1},
                {"name": "t3", "shape": [2048, 2048], "dtype": "float32", "shard_id": 2},
                {"name": "t4", "shape": [2048, 2048], "dtype": "float32", "shard_id": 2},
            ],
            lazy=False,
        ),
    )
]
