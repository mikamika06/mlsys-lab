def should_exclude(module_name: str, exclude_patterns: list[str]) -> bool:
    """Check if a module should be excluded from sensitivity profiling."""
    raise NotImplementedError


def filter_modules(modules: list[dict], exclude_patterns: list[str]) -> list[dict]:
    """Return modules that are not excluded by pattern rules."""
    raise NotImplementedError


def profile_sensitivities(
    modules: list[dict],
    candidate_bits: list[int],
    exclude_patterns: list[str],
    eval_fn: callable,
) -> dict[str, dict[int, float]]:
    """Build a sensitivity profile mapping module names to bit-width error costs."""
    raise NotImplementedError
