"""VRAM estimation functions."""


def vram_expected_gb(vanilla_peak_gb: float, published_pct_savings: float) -> float:
    """Calculate expected VRAM usage in GB given baseline VRAM and percentage savings claim."""
    if vanilla_peak_gb < 0.0 or published_pct_savings < 0.0 or published_pct_savings > 100.0:
        raise ValueError("Invalid parameters for VRAM calculation")
    return round(vanilla_peak_gb * (1.0 - published_pct_savings / 100.0), 4)
