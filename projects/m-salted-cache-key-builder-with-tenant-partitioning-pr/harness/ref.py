import hashlib

TEST_SCENARIOS = [
    {
        "tenant_id": "tenant_alpha",
        "salt": "v1_salt",
        "block_size": 4,
        "tokens": [10, 20, 30, 40, 50, 60, 70, 80, 90],
    },
    {
        "tenant_id": "tenant_beta",
        "salt": "v1_salt",
        "block_size": 4,
        "tokens": [10, 20, 30, 40, 50, 60, 70, 80, 90],
    },
    {
        "tenant_id": "tenant_alpha",
        "salt": "v2_salt",
        "block_size": 2,
        "tokens": [1, 2, 3, 4, 5, 6],
    },
]

TTFT_BENCHMARKS = {
    "baseline_ttft_per_token": 0.005,
    "cache_hit_ttft": 0.002,
    "samples": [
        {"prompt_len": 128, "ttft": 0.642},
        {"prompt_len": 128, "ttft": 0.322},
        {"prompt_len": 256, "ttft": 0.642},
        {"prompt_len": 512, "ttft": 2.562},
    ],
}

PAIR_TEST_CASES = [
    (
        {"tenant_id": "t1", "salt": "s1"},
        {"tenant_id": "t1", "salt": "s1"},
        False,
    ),
    (
        {"tenant_id": "t1", "salt": "s1"},
        {"tenant_id": "t2", "salt": "s1"},
        False,
    ),
    (
        {"tenant_id": "t1", "salt": "s1"},
        {"tenant_id": "t2", "salt": "s1"},
        True,
    ),
    (
        {"tenant_id": "t1", "salt": "s1"},
        {"tenant_id": "t1", "salt": "s2"},
        True,
    ),
]


def compute_block_hash(tenant_id, block_tokens, salt="", parent_hash=""):
    hasher = hashlib.sha256()
    hasher.update(str(tenant_id).encode("utf-8"))
    hasher.update(b":")
    hasher.update(str(salt).encode("utf-8"))
    hasher.update(b":")
    hasher.update(str(parent_hash).encode("utf-8"))
    hasher.update(b":")
    for tok in block_tokens:
        hasher.update(int(tok).to_bytes(4, byteorder="big", signed=True))
        hasher.update(b",")
    return hasher.hexdigest()


def build_prefix_keys(tenant_id, tokens, block_size, salt=""):
    if block_size <= 0:
        return []
    keys = []
    parent_hash = ""
    num_blocks = len(tokens) // block_size
    for i in range(num_blocks):
        block_tokens = tokens[i * block_size : (i + 1) * block_size]
        h = compute_block_hash(tenant_id, block_tokens, salt=salt, parent_hash=parent_hash)
        keys.append(h)
        parent_hash = h
    return keys


def infer_prefix_residency(ttft_samples, baseline_ttft_per_token, cache_hit_ttft):
    res = []
    for sample in ttft_samples:
        prompt_len = sample["prompt_len"]
        measured_ttft = sample["ttft"]
        saved_time = max(0.0, (prompt_len * baseline_ttft_per_token + cache_hit_ttft) - measured_ttft)
        tokens_cached = int(round(saved_time / baseline_ttft_per_token)) if baseline_ttft_per_token > 0 else 0
        tokens_cached = max(0, min(prompt_len, tokens_cached))
        res.append(tokens_cached)
    return res


def can_share_blocks(req_a, req_b, allow_cross_tenant=False):
    tenant_a = req_a.get("tenant_id")
    tenant_b = req_b.get("tenant_id")
    if not allow_cross_tenant and tenant_a != tenant_b:
        return False
    salt_a = req_a.get("salt", "")
    salt_b = req_b.get("salt", "")
    if salt_a != salt_b:
        return False
    return True
