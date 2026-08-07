import hashlib
import numpy as np

BLOCK_SIZE = 16

REQUESTS = [
    {
        "tenant_id": "tenant_alpha",
        "tenant_salt": "salt_alpha_sec_9081",
        "tokens": [101, 2054, 2003, 1037, 3829, 2000, 1037, 2742, 1010, 2000, 1037, 2742, 1012, 102, 0, 0] * 4
    },
    {
        "tenant_id": "tenant_beta",
        "tenant_salt": "salt_beta_sec_1102",
        "tokens": [101, 2054, 2003, 1037, 3829, 2000, 1037, 2742, 1010, 2000, 1037, 2742, 1012, 102, 0, 0] * 4
    },
    {
        "tenant_id": "tenant_gamma",
        "tenant_salt": "salt_gamma_sec_3341",
        "tokens": [101, 2054, 2003, 1037, 3829, 2000, 1037, 2742, 1010, 2000, 1037, 2742, 1012, 102, 0, 0] * 4
    }
]

np.random.seed(42)
TTFT_HITS = list(np.random.normal(loc=12.5, scale=1.2, size=50))
TTFT_MISSES = list(np.random.normal(loc=85.0, scale=4.5, size=50))

LAUNCH_CONFIGS = [
    {
        "name": "prod_config_1",
        "flags": {"enable_prefix_caching": True, "tenant_salt_enabled": True},
        "env": {"VLLM_ALLOW_CROSS_TENANT_CACHE": "0"}
    },
    {
        "name": "prod_config_2",
        "flags": {"enable_prefix_caching": True, "tenant_salt_enabled": False},
        "env": {"VLLM_ALLOW_CROSS_TENANT_CACHE": "0"}
    },
    {
        "name": "prod_config_3",
        "flags": {"enable_prefix_caching": True, "shared_cache_salt": True},
        "env": {"VLLM_ALLOW_CROSS_TENANT_CACHE": "0"}
    },
    {
        "name": "prod_config_4",
        "flags": {"enable_prefix_caching": False, "tenant_salt_enabled": False},
        "env": {"VLLM_ALLOW_CROSS_TENANT_CACHE": "1"}
    },
    {
        "name": "prod_config_5",
        "flags": {"enable_prefix_caching": True, "tenant_salt_enabled": True, "enable_profiling": True, "expose_metrics_publicly": True},
        "env": {"VLLM_ALLOW_CROSS_TENANT_CACHE": "0"}
    },
    {
        "name": "prod_config_6",
        "flags": {"enable_prefix_caching": True, "tenant_salt_enabled": True},
        "env": {"VLLM_ALLOW_CROSS_TENANT_CACHE": "0"}
    },
    {
        "name": "prod_config_7",
        "flags": {"enable_prefix_caching": True, "tenant_salt_enabled": False, "shared_cache_salt": True},
        "env": {"VLLM_ALLOW_CROSS_TENANT_CACHE": "1"}
    },
    {
        "name": "prod_config_8",
        "flags": {"enable_prefix_caching": False},
        "env": {"VLLM_ALLOW_CROSS_TENANT_CACHE": "0"}
    },
    {
        "name": "prod_config_9",
        "flags": {"enable_prefix_caching": True, "tenant_salt_enabled": True, "enable_profiling": True, "expose_metrics_publicly": False},
        "env": {"VLLM_ALLOW_CROSS_TENANT_CACHE": "0"}
    },
    {
        "name": "prod_config_10",
        "flags": {"enable_prefix_caching": True, "tenant_salt_enabled": True},
        "env": {"VLLM_ALLOW_CROSS_TENANT_CACHE": "0"}
    }
]


def compute_block_hashes(tokens, block_size, tenant_salt):
    hashes = []
    num_blocks = len(tokens) // block_size
    prefix_hash = ""
    for b in range(num_blocks):
        block_tokens = tokens[b * block_size: (b + 1) * block_size]
        payload = f"{prefix_hash}:{tenant_salt}:{','.join(map(str, block_tokens))}"
        h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        hashes.append(h)
        prefix_hash = h
    return hashes


def check_tenant_isolation(requests, block_size):
    tenant_blocks = {}
    shared_count = 0
    for req in requests:
        tenant = req["tenant_id"]
        salt = req["tenant_salt"]
        tokens = req["tokens"]
        hashes = compute_block_hashes(tokens, block_size, salt)
        if tenant not in tenant_blocks:
            tenant_blocks[tenant] = set()
        for h in hashes:
            tenant_blocks[tenant].add(h)

    all_tenants = list(tenant_blocks.keys())
    for i in range(len(all_tenants)):
        for j in range(i + 1, len(all_tenants)):
            t1, t2 = all_tenants[i], all_tenants[j]
            overlap = tenant_blocks[t1].intersection(tenant_blocks[t2])
            shared_count += len(overlap)

    is_isolated = (shared_count == 0)
    return is_isolated, shared_count


def quantify_ttft_side_channel(ttft_hits, ttft_misses):
    hits = np.array(ttft_hits, dtype=np.float64)
    misses = np.array(ttft_misses, dtype=np.float64)
    mean_hit = float(np.mean(hits))
    mean_miss = float(np.mean(misses))
    delta_ttft = float(mean_miss - mean_hit)
    std_hit = float(np.std(hits, ddof=1)) if len(hits) > 1 else 0.0
    std_miss = float(np.std(misses, ddof=1)) if len(misses) > 1 else 0.0
    n_hit = len(hits)
    n_miss = len(misses)
    pooled_std = float(np.sqrt(((n_hit - 1) * (std_hit ** 2) + (n_miss - 1) * (std_miss ** 2)) / (n_hit + n_miss - 2)))
    cohens_d = float(delta_ttft / pooled_std) if pooled_std > 0 else 0.0
    p95_hit = float(np.percentile(hits, 95))
    p5_miss = float(np.percentile(misses, 5))
    separable = float(p95_hit < p5_miss)
    return {
        "mean_hit_ms": mean_hit,
        "mean_miss_ms": mean_miss,
        "delta_ttft_ms": delta_ttft,
        "cohens_d": cohens_d,
        "distinguishable": separable
    }


def audit_launch_configs(configs):
    results = []
    for cfg in configs:
        name = cfg.get("name", "unknown")
        flags = cfg.get("flags", {})
        env = cfg.get("env", {})
        issues = []
        if flags.get("enable_prefix_caching", False) and not flags.get("tenant_salt_enabled", False):
            issues.append("prefix_caching_without_tenant_salt")
        if flags.get("shared_cache_salt", False):
            issues.append("shared_cache_salt_across_tenants")
        if env.get("VLLM_ALLOW_CROSS_TENANT_CACHE", "0") == "1":
            issues.append("cross_tenant_cache_env_enabled")
        if flags.get("enable_profiling", False) and flags.get("expose_metrics_publicly", False):
            issues.append("exposed_profiling_side_channel")
        is_secure = (len(issues) == 0)
        results.append({
            "name": name,
            "secure": is_secure,
            "issues": issues
        })
    return results
