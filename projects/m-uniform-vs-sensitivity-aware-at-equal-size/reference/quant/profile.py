def should_exclude(module_name: str, exclude_patterns: list[str]) -> bool:
    """Check if a module should be excluded from sensitivity profiling."""
    for pattern in exclude_patterns:
        if pattern in module_name:
            return True
    return False


def filter_modules(modules: list[dict], exclude_patterns: list[str]) -> list[dict]:
    """Return modules that are not excluded by pattern rules."""
    return [m for m in modules if not should_exclude(m["name"], exclude_patterns)]


def profile_sensitivities(
    modules: list[dict],
    candidate_bits: list[int],
    exclude_patterns: list[str],
    eval_fn: callable,
) -> dict[str, dict[int, float]]:
    """Build a sensitivity profile mapping module names to bit-width error costs."""
    profile = {}
    included = filter_modules(modules, exclude_patterns)
    for mod in included:
        m_name = mod["name"]
        profile[m_name] = {}
        for b in candidate_bits:
            profile[m_name][b] = float(eval_fn(m_name, b))
    return profile
