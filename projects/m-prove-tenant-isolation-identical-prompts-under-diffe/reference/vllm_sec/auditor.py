from typing import List, Dict, Any


def audit_launch_configs(configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
