def derived_mac_per_cycle(is_vnni: bool, vector_width: int) -> float:
    """Return theoretical MAC operations per cycle."""
    raise NotImplementedError


def analyze_vnni_vs_fallback(
    num_mac_ops: int,
    vnni_vector_width: int = 512,
    fallback_vector_width: int = 256,
    base_freq_ghz: float = 3.0,
    recovery_cycles: int = 50000,
) -> dict:
    """Analyze VNNI speedup vs fallback including downclocking penalty."""
    raise NotImplementedError
